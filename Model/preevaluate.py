import sys
import json
import torch
from pathlib import Path
from torch.nn import MSELoss
from model import AnatomyModel
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/mrtcloud-1/Documents/Hand-Tracking-2/Dataset")
from optimizedataset import DatasetOptimizer

with open(
    "/Users/michaeltoppin/Documents/Coding/Hand-Tracking-2/Model/configs.json", "r"
) as f:
    configs = json.load(f)

chol_row = torch.load(Path(configs["post_data_dir"]) / "cholrow.pt")
chol_col = torch.load(Path(configs["post_data_dir"]) / "cholcol.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
features = DatasetOptimizer(mp=False).getFeatures
criterion = MSELoss()


def evalModel():
    decoder_test_loss = 0
    generator_test_loss = 0
    gen_samples = 0

    # Decoder
    with torch.inference_mode():
        for decoder_batch in decoder_test_loader:
            decoder_batch = decoder_batch.to(device)
            decoder_coords = decoder_batch.x.float()
            decoder_edge_index = decoder_batch.edge_index
            decoder_b = decoder_batch.batch

            normalized_coords, _, _ = features(
                decoder_coords.detach().cpu().numpy(), None, True
            )  # Original 3D normalized features
            normalized_coords = torch.tensor(normalized_coords).to(device)

            with torch.inference_mode():
                mean_mat, scale_col, scale_row = generator_model(
                    normalized_coords, decoder_edge_index, decoder_b
                )
                mean_mat = mean_mat.reshape(21, 3)
                Z = torch.rand_like(mean_mat)
                noise = mean_mat + torch.sqrt(scale_col * scale_row) * (
                    chol_row @ Z @ chol_col.T
                )

                normalized_coords, coords_proj, scale = features(
                    decoder_coords.detach().cpu().numpy(), noise, True
                )  # Distorted 3D normalized features with noise
                normalized_coords = torch.as_tensor(
                    normalized_coords, dtype=torch.float32, device=device
                )
                coords_proj = torch.as_tensor(
                    coords_proj, dtype=torch.float32, device=device
                )
                scale = torch.as_tensor(scale, dtype=torch.float32, device=device)

                errors = decoder_model(normalized_coords, decoder_edge_index, decoder_b)
                pred_coords = coords_proj + (scale * errors)
                decoder_loss = criterion(pred_coords, decoder_coords)
                decoder_test_loss += decoder_loss.item()
    decoder_test_loss /= len(decoder_test_loader)

    # Generator
    with torch.inference_mode():
        for generator_batch, gen_targets, gen_raw in generator_test_loader:
            generator_features = generator_batch.x.float()
            generator_edge_index = generator_batch.edge_index
            generator_b = generator_batch.batch

            gen_scale = torch.linalg.norm(gen_targets[:, 9] - gen_targets[:, 0])
            true_errors = (gen_targets - gen_raw) / gen_scale

            pred_errors = generator_model(
                generator_features, generator_edge_index, generator_b
            )
            generator_loss = criterion(pred_errors, true_errors)
            batch_size = gen_targets.size(0)
            generator_test_loss += generator_loss.item() * batch_size
            gen_samples += batch_size
    generator_test_loss /= gen_samples

    return decoder_test_loss, generator_test_loss


evaluation = evalModel()
print(
    f"Average Decoder Loss: {evaluation[0]: .5f} | Average Generator Loss: {evaluation[1]: .5f}"
)
