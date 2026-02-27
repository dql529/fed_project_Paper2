import os, random, copy, time, math
import torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# --- Set Seeds for Reproducibility ---
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# --- Model Definitions ---
class StudentNet(nn.Module):
    def __init__(self, num_output_features):
        super(StudentNet, self).__init__()
        self.num_output_features = num_output_features
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, num_output_features)

    def forward(self, x, edge_index=None):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TeacherNet(nn.Module):
    def __init__(self, num_output_features):
        super(TeacherNet, self).__init__()
        self.conv1 = GCNConv(9, 27)
        self.conv2 = GCNConv(27, 27)
        self.fc = nn.Linear(27, num_output_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x


# --- Helper & Aggregation Functions ---
def weighted_average_aggregation(models, reputations):
    total_rep = sum(reputations.values()) or 1.0
    agg_dict = {}
    for i, model in enumerate(models):
        weight = reputations.get(model.drone_id, 0.0) / total_rep
        for k, v in model.state_dict().items():
            if i == 0:
                agg_dict[k] = v.clone() * weight
            else:
                agg_dict[k] += v * weight
    return agg_dict


def model_params_similarity_R1(local_dict, ref_dict):
    if not ref_dict:
        return 0.5
    diff = sum(
        torch.mean(torch.abs(local_dict[k] - ref_dict[k]))
        for k in local_dict
        if k in ref_dict
    )
    return 1.0 / (1.0 + diff / len(local_dict)) if len(local_dict) > 0 else 0.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def exponential_decay(x, a=0.8):
    return np.exp(-a * x)


# --- Unified Data Loading and Teacher Training ---
def load_data_and_train_teacher(num_nodes):
    print("--- Loading and Preparing Unified Dataset ---")
    try:
        df = pd.read_csv("UAV9-main/UAVCAN/type10_scaled.csv")
    except FileNotFoundError:
        print("ERROR: 'UAV9-main/UAVCAN/type10_scaled.csv' not found. Exiting.")
        exit()

    adjacency_matrix = torch.tensor(
        [
            [0, 1, 1, 0, 1, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.float,
    )
    edge_index = adjacency_matrix.nonzero(as_tuple=False).t()

    train_pool_df, test_df = train_test_split(df, test_size=0.1, random_state=0)

    global_test_data = Data(
        x=torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(test_df.iloc[:, -1].values, dtype=torch.long),
    )

    node_datasets = []
    indices = np.arange(len(train_pool_df))
    np.random.shuffle(indices)
    split_size = len(train_pool_df) // num_nodes
    for i in range(num_nodes):
        start = i * split_size
        end = None if i == num_nodes - 1 else (i + 1) * split_size
        node_indices = indices[start:end]
        node_datasets.append(train_pool_df.iloc[node_indices])

    node_data_objects = []
    for node_df in node_datasets:
        train_node_df, test_node_df = train_test_split(
            node_df, test_size=0.2, random_state=0
        )
        train_obj = Data(
            x=torch.tensor(train_node_df.iloc[:, :-1].values, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(train_node_df.iloc[:, -1].values, dtype=torch.long),
        )
        test_obj = Data(
            x=torch.tensor(test_node_df.iloc[:, :-1].values, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(test_node_df.iloc[:, -1].values, dtype=torch.long),
        )
        node_data_objects.append({"train": train_obj, "test": test_obj})

    if not os.path.exists("large_model.pt"):
        print("--- Training Teacher Model (large_model.pt) ---")
        teacher_model = TeacherNet(num_output_features=2)
        optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        best_accuracy = 0.0
        best_model_state_dict = None

        # Use DataLoader ONLY for teacher training to manage memory
        teacher_train_data = Data(
            x=torch.tensor(train_pool_df.iloc[:, :-1].values, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(train_pool_df.iloc[:, -1].values, dtype=torch.long),
        )
        # PyG's DataLoader can handle a single large graph object
        train_loader = DataLoader(
            [teacher_train_data], batch_size=1
        )  # Batch size of 1 for full graph training

        for epoch in range(100):
            teacher_model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = teacher_model(batch.x, batch.edge_index)
                loss = criterion(outputs, batch.y)
                loss.backward()
                optimizer.step()

            teacher_model.eval()
            with torch.no_grad():
                test_outputs = teacher_model(
                    global_test_data.x, global_test_data.edge_index
                )
                preds = test_outputs.argmax(dim=1)
                acc = accuracy_score(global_test_data.y.cpu(), preds.cpu())
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model_state_dict = copy.deepcopy(teacher_model.state_dict())
            if (epoch + 1) % 20 == 0:
                print(
                    f"Teacher Training Epoch {epoch+1}/100, Best Acc on Global Test: {best_accuracy:.4f}"
                )

        torch.save(best_model_state_dict, "large_model.pt")
        print(
            f"--- Teacher Model training finished. Saved to large_model.pt (Best Acc: {best_accuracy:.4f}) ---"
        )

    return global_test_data, node_data_objects


# --- Simulation Classes ---
class DroneNode:
    def __init__(self, drone_id, is_malicious=False):
        self.drone_id = drone_id
        self.is_malicious = is_malicious
        self.local_model = None
        self.data = None
        self.test_data = None
        self.num_epochs = 2
        self.learning_rate = 0.014

    def receive_global_model(self, global_model):
        self.local_model = copy.deepcopy(global_model)

    def train(self):
        if not self.local_model or self.data is None:
            return
        optimizer = torch.optim.Adam(
            self.local_model.parameters(), lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        best_accuracy = -1.0
        best_model_state_dict = None
        for _ in range(self.num_epochs):
            self.local_model.train()
            optimizer.zero_grad()
            outputs = self.local_model(self.data.x, self.data.edge_index)
            loss = criterion(outputs, self.data.y.squeeze().long())
            loss.backward()
            optimizer.step()

            self.local_model.eval()
            with torch.no_grad():
                test_outputs = self.local_model(
                    self.test_data.x, self.test_data.edge_index
                )
                preds = test_outputs.argmax(dim=1)
                acc = accuracy_score(self.test_data.y.cpu(), preds.cpu())
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_model_state_dict = copy.deepcopy(self.local_model.state_dict())
        if best_model_state_dict:
            self.local_model.load_state_dict(best_model_state_dict)
        if self.is_malicious:
            with torch.no_grad():
                for param in self.local_model.parameters():
                    param.add_(torch.randn(param.size()) * 0.1)


class CentralServer:
    def __init__(self, ablation_config="R1,R2,R3,R4"):
        self.global_model = StudentNet(num_output_features=2)
        self.teacher_model = TeacherNet(num_output_features=2)
        self.teacher_model.load_state_dict(
            torch.load("large_model.pt", map_location=torch.device("cpu"))
        )
        self.teacher_model.eval()
        self.reputation = {}
        self.data_age = {}
        self.ablation_config = ablation_config
        self.reputation_log = []
        self.low_performance_counts = {}
        self.last_broadcast_state = None

    def aggregate_models(self, local_models):
        for i, model in enumerate(local_models):
            model.drone_id = i
        aggregated_dict = weighted_average_aggregation(local_models, self.reputation)
        self.global_model.load_state_dict(aggregated_dict)
        self.last_broadcast_state = copy.deepcopy(self.global_model.state_dict())

    def compute_r4(self, local_model_output, large_model_output, temperature=1.0):
        # Confidence-gated JS similarity: only high-confidence teacher outputs participate.
        confidence_thresh = 0.8
        with torch.no_grad():
            softmax = torch.nn.Softmax(dim=1)
            local_probs = softmax(local_model_output / temperature)
            large_probs = softmax(large_model_output / temperature)

            teacher_conf, _ = large_probs.max(dim=1)
            mask = (teacher_conf >= confidence_thresh).float()
            if mask.sum() < 1:
                return 0.5  # No reliable teacher predictions, return neutral value.

            m = 0.5 * (local_probs + large_probs)
            # Per-sample JS divergence
            js_per_sample = 0.5 * (
                F.kl_div(torch.log(local_probs + 1e-8), m, reduction="none").sum(dim=1)
                + F.kl_div(torch.log(large_probs + 1e-8), m, reduction="none").sum(
                    dim=1
                )
            )
            js_div = (mask * js_per_sample).sum() / mask.sum()
            js_sim = 1 - js_div.item()
            return max(0, js_sim)

    def compute_reputation(
        self,
        drone_id,
        local_model,
        performance,
        data_age,
        local_model_output,
        large_model_output,
        current_round,
    ):
        active_r = set(self.ablation_config.split(","))

        R1 = (
            model_params_similarity_R1(
                local_model.state_dict(), self.last_broadcast_state
            )
            if "R1" in active_r
            else 0.5
        )
        R2 = sigmoid(performance) if "R2" in active_r else 0.5
        R3 = exponential_decay(data_age) if "R3" in active_r else 0.5
        R4 = (
            self.compute_r4(local_model_output, large_model_output)
            if "R4" in active_r
            else 0.5
        )

        S1 = math.exp(R1 - 0.5)
        S2 = math.exp(R2 - 0.5)
        S3 = math.exp(R3 - 0.5)
        S4 = math.exp(R4 - 0.5)
        total_confidence = S1 + S2 + S3 + S4 + 1e-8
        W1 = S1 / total_confidence
        W2 = S2 / total_confidence
        W3 = S3 / total_confidence
        W4 = S4 / total_confidence

        reputation = W1 * R1 + W2 * R2 + W3 * R3 + W4 * R4

        if performance < 0.65:
            self.low_performance_counts[drone_id] = (
                self.low_performance_counts.get(drone_id, 0) + 1
            )
            reputation *= 0.1
        else:
            self.low_performance_counts[drone_id] = 0

        self.reputation_log.append(
            {
                "Round": current_round,
                "Drone_ID": drone_id,
                "R1": R1,
                "R2": R2,
                "R3": R3,
                "R4": R4,
                "Final_Rep": reputation,
            }
        )
        return reputation


# --- Main Simulation Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    ABLATION_CONFIGS = {
        "FullModel_R1234": "R1,R2,R3,R4",
        "No_R1": "R2,R3,R4",
        "No_R2": "R1,R3,R4",
        "No_R3": "R1,R2,R4",
        "No_R4": "R1,R2,R3",
        "Only_R1": "R1",
        "Only_R2": "R2",
        "Only_R3": "R3",
        "Only_R4": "R4",
    }
    ACTIVE_ABLATION = "No_R1"
    NUM_NODES = 5
    MALICIOUS_NODES = 2
    NUM_ROUNDS = 20
    CHECKPOINTS = [5, 10, 15, 20]
    RESULTS = []

    global_test_data, node_data_objects = load_data_and_train_teacher(NUM_NODES)

    print(f"Starting Simulation v2 for: {ACTIVE_ABLATION}")
    server = CentralServer(ablation_config=ABLATION_CONFIGS[ACTIVE_ABLATION])
    nodes = [
        DroneNode(drone_id=i, is_malicious=(i < MALICIOUS_NODES))
        for i in range(NUM_NODES)
    ]

    for i, node in enumerate(nodes):
        node.data = node_data_objects[i]["train"]
        node.test_data = node_data_objects[i]["test"]
        server.data_age[node.drone_id] = 0
        server.reputation[node.drone_id] = 1.0

    for r in range(1, NUM_ROUNDS + 1):
        print(f"--- Round {r} ---")
        local_models_for_aggregation = []
        node_performances = {}
        local_model_outputs = {}

        with torch.no_grad():
            large_model_output = server.teacher_model(
                global_test_data.x, global_test_data.edge_index
            )

        for node in nodes:
            node.receive_global_model(server.global_model)
            node.train()
            local_models_for_aggregation.append(node.local_model)
            with torch.no_grad():
                outputs = node.local_model(
                    global_test_data.x, global_test_data.edge_index
                )
                local_model_outputs[node.drone_id] = outputs
                preds = outputs.argmax(dim=1)
                node_performances[node.drone_id] = accuracy_score(
                    global_test_data.y.cpu(), preds.cpu()
                )

        for node in nodes:
            server.reputation[node.drone_id] = server.compute_reputation(
                node.drone_id,
                node.local_model,
                node_performances[node.drone_id],
                server.data_age[node.drone_id],
                local_model_outputs[node.drone_id],
                large_model_output,
                r,
            )
            server.data_age[node.drone_id] += 1

        server.aggregate_models(local_models_for_aggregation)

        if r in CHECKPOINTS:
            print(f"Checkpoint at round {r}: Evaluating global model...")
            with torch.no_grad():
                outputs = server.global_model(
                    global_test_data.x, global_test_data.edge_index
                )
                preds = outputs.argmax(dim=1)
                acc = accuracy_score(global_test_data.y.cpu(), preds.cpu())
                f1 = f1_score(global_test_data.y.cpu(), preds.cpu())
                print(f"Global Model Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
                RESULTS.append(
                    {
                        "Round": r,
                        "Config": ACTIVE_ABLATION,
                        "Accuracy": acc,
                        "F1_Score": f1,
                    }
                )

    print("\n--- Simulation Finished ---")
    results_df = pd.DataFrame(RESULTS)
    print(results_df)
    results_df.to_csv(f"simulation_v2_results_{ACTIVE_ABLATION}.csv", index=False)
    print(f"Results saved to simulation_v2_results_{ACTIVE_ABLATION}.csv")

    reputation_df = pd.DataFrame(server.reputation_log)
    if not reputation_df.empty:
        reputation_df.to_csv(f"reputation_v2_log_{ACTIVE_ABLATION}.csv", index=False)
        print(f"Reputation log saved to reputation_v2_log_{ACTIVE_ABLATION}.csv")
