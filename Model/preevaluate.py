import json
import torch
from pathlib import Path
from torch.nn import MSELoss
from model import AnatomyModel
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from pretrain import generator_criterion as gc
from pretrainer import Trainer

with open(
    "/Users/michaeltoppin/Documents/Coding/Hand-Tracking-2/Model/configs.json", "r"
) as f:
    configs = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for eval sampling reproducability
SEED = 42
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

decoder_model = AnatomyModel(
    configs["input_size"],
    configs["decoder_hidden_size"],
    configs["decoder_output_size"],
    configs["decoder_dropout"],
    generator=False,
).to(device)

decoder_weights = torch.load(
    (Path(configs["model_dir"]) / configs["decoder_model_name"]),
    weights_only=True,
    map_location=device,
)

decoder_model.load_state_dict(decoder_weights)
decoder_model.eval()

generator_model = AnatomyModel(
    configs["input_size"],
    configs["generator_hidden_size"],
    configs["generator_output_size"],
    configs["generator_dropout"],
    generator=True,
).to(device)

generator_weights = torch.load(
    (Path(configs["model_dir"]) / configs["generator_model_name"]),
    weights_only=True,
    map_location=device,
)

generator_model.load_state_dict(generator_weights)
generator_model.eval()


def collateDecoder(batch):
    coords, _ = zip(*batch)
    coord_batch = Batch.from_data_list(list(coords))
    return coord_batch


def collateGenerator(batch):
    graphs, targets, raw_coords = zip(*batch)
    batch = Batch.from_data_list(list(graphs))
    targets = torch.stack(targets, dim=0).float()
    raw_coords = torch.stack(raw_coords, dim=0).float()
    return (batch, targets, raw_coords)


decoder_test_dataset = torch.load(Path(configs["data_dir"]) / "Testing" / "dataset.pt")
decoder_test_loader = DataLoader(
    dataset=decoder_test_dataset,
    num_workers=configs["num_workers"],
    batch_size=configs["batch_size"],
    drop_last=configs["drop_last"],
    collate_fn=collateDecoder,
)

generator_test_dataset = torch.load(
    Path(configs["post_data_dir"]) / "Testing" / "dataset.pt"
)
generator_test_loader = DataLoader(
    dataset=generator_test_dataset,
    num_workers=configs["num_workers"],
    batch_size=configs["batch_size"],
    drop_last=configs["drop_last"],
    collate_fn=collateGenerator,
)

features = Trainer().features
decoder_criterion = MSELoss()
generator_criterion = gc

chol_row = torch.load(
    Path(configs["post_data_dir"]) / "cholrow.pt",
    weights_only=False,
).to(device)

chol_col = torch.load(
    Path(configs["post_data_dir"]) / "cholcol.pt",
    weights_only=False,
).to(device)

cov_row = chol_row @ chol_row.T
cov_col = chol_col @ chol_col.T

col_inv = torch.linalg.inv(cov_col)
logdet_col = torch.linalg.slogdet(cov_col).logabsdet


def evalModel():
    decoder_test_loss = 0
    generator_test_loss = 0
    dec_samples = 0
    gen_samples = 0

    # Decoder Validation
    with torch.inference_mode():
        for decoder_batch in decoder_test_loader:
            decoder_batch = decoder_batch.to(device)
            decoder_coords = decoder_batch.x.float()
            decoder_edge_index = decoder_batch.edge_index
            decoder_b = decoder_batch.batch

            normalized_coords, _, _ = features(decoder_coords, None)
            mean_mat, scale = generator_model(
                normalized_coords, decoder_edge_index, decoder_b
            )
            mean_mat = mean_mat.reshape(decoder_batch.num_graphs, 21, 3)
            Z = torch.randn_like(mean_mat)
            noise = mean_mat + torch.sqrt(scale)[:, None, None] * (
                chol_row @ Z @ chol_col.T
            )

            normalized_coords, coords_proj, scale = features(decoder_coords, noise)
            errors = decoder_model(normalized_coords, decoder_edge_index, decoder_b)
            pred_coords = coords_proj + (scale * errors)
            decoder_loss = decoder_criterion(pred_coords, decoder_coords)
            decoder_test_loss += decoder_loss.item() * decoder_batch.num_graphs
            dec_samples += decoder_batch.num_graphs
        decoder_test_loss /= dec_samples

        # Generator Validation
        with torch.inference_mode():
            for generator_batch, gen_targets, gen_raw in generator_test_loader:
                generator_features = generator_batch.x.float()
                generator_edge_index = generator_batch.edge_index
                generator_b = generator_batch.batch

                gen_scale = torch.linalg.norm(
                    gen_targets[:, 9] - gen_targets[:, 0], dim=1
                )
                true_errors = (gen_targets - gen_raw) / gen_scale[:, None, None]

                mean_mat, scale = generator_model(
                    generator_features, generator_edge_index, generator_b
                )
                mean_mat = mean_mat.reshape(generator_batch.num_graphs, 21, 3)
                generator_loss = generator_criterion(
                    true_errors,
                    mean_mat,
                    scale,
                    cov_row,
                    col_inv,
                    logdet_col,
                    device,
                )
                generator_test_loss += (
                    generator_loss.item() * generator_batch.num_graphs
                )
                gen_samples += generator_batch.num_graphs
        generator_test_loss /= gen_samples

    return decoder_test_loss, generator_test_loss


evaluation = evalModel()
print(
    f"Average Decoder Loss: {evaluation[0]: .5f} | Average Generator Loss: {evaluation[1]: .5f}"
)
