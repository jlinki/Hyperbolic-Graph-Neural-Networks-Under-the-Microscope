"""Base model class."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.encoders as encoders
from models.decoders import model2decoder

from sklearn.metrics import f1_score as compute_f1_score
from sklearn.metrics import roc_auc_score, average_precision_score

from abc import ABC, abstractmethod

from models.utils_models import proj_tan0, make_manifold

import wandb


class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, scores):
        probs = 1. / (torch.exp(((-scores - self.r) / self.t).clamp_max(20)) + 1.0)
        return probs


class MarginLoss(torch.nn.Module):
    def __init__(self, margin) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, preds, data_edge_label=None):
        pos_scores = preds[data_edge_label==1.0]
        neg_scores = preds[data_edge_label==0.0]
        return torch.nn.functional.relu(self.margin - pos_scores[:, None] + neg_scores[None, :]).mean()



class StressLoss(nn.Module):
    """
    Stress loss for pairwise distances (1D tensors).
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, D_pred: torch.Tensor, D_true: torch.Tensor) -> torch.Tensor:
        # D_pred, D_true: shape [num_pairs]
        weights = 1.0 / (D_true ** 2 + self.eps)
        loss = (weights * (D_pred - D_true) ** 2).mean()
        return loss


class StressLossNR(nn.Module):
    """
    Stress loss for pairwise distances (1D tensors) ignoring the target 0 node.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, D_pred: torch.Tensor, D_true: torch.Tensor) -> torch.Tensor:
        # D_pred, D_true: shape [num_pairs]
        mask = torch.where(D_true > 0)[0]
        weights = 1.0 / (D_true ** 2 + self.eps)
        loss = (weights[mask] * (D_pred[mask] - D_true[mask]) ** 2).mean()
        return loss



class BCELoss(torch.nn.Module):
    def __init__(self, decoder) -> None:
        super().__init__()
        self.dc_function = decoder
        self.loss = torch.nn.BCELoss()
    def forward(self, preds, data_edge_label=None):
        return self.loss(self.dc_function(preds), data_edge_label)



class BaseModel(nn.Module, ABC):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.manifold_name = args.manifold
        self.modelname = args.model
        self.exp_map_clip = args.exp_map_clip

        self.encoder = getattr(encoders, args.model)(args)

        if self.modelname.lower() == 'hgcn':
            self.manifold = self.encoder.manifolds[0]
        else:
            self.manifold = make_manifold(self.manifold_name, args.c)
             

    def encode(self, x, edge_index, edge_attr, return_all_layers=False):
        if self.manifold_name in ['Lorentz']:
            time_coord = torch.zeros((x.shape[0],1), device=x.device)
            xt = torch.cat((time_coord,x), dim=1) # tangent space of origin
            x = self.manifold.expmap0(xt)
            if self.exp_map_clip is not None:
                if self.exp_map_clip != "None":
                    # remove nans
                    x = torch.nan_to_num(x, nan=1e20)
                    # normalize to 1e9
                    x = x.clamp_min(-float(self.exp_map_clip)).clamp_max(float(self.exp_map_clip))
        elif self.manifold_name == 'PoincareBall':
            xt = x
            if self.exp_map_clip not in [None, "None"]:
                xt = xt.clamp(-self.exp_map_clip, self.exp_map_clip)
            x = self.manifold.expmap0(xt)
            x = self.manifold.projx(x)
            if self.exp_map_clip is not None:
                if self.exp_map_clip != "None":
                    # remove nans
                    x = torch.nan_to_num(x, nan=1e20)
                    # normalize to 1e9
                    x = x.clamp_min(-float(self.exp_map_clip)).clamp_max(float(self.exp_map_clip))
        h, embedding_list = self.encoder.encode(x, edge_index, edge_attr, return_all_layers)

        return h, embedding_list
    
    @abstractmethod
    def compute_loss(self, exp, embeddings, data, split, mode):
        pass
    
    @abstractmethod
    def compute_metrics(self, exp, embeddings, data, split):
        pass
    
    @abstractmethod
    def which_metrics(self):
        pass
    
    @abstractmethod
    def init_best_metric(self, key):
        pass

    @abstractmethod
    def has_improved(self, m1, m2, key):
        pass

    @abstractmethod
    def wandb_log(self, exp, epoch, split, metrics, best_eval=False):
        pass


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        if self.manifold_name in ['Lorentz', 'PoincareBall']:
            self.decoder = model2decoder[args.model](args)
        else:
            if args.model.lower() == 'hybonet':
                self.decoder = model2decoder['MLP'](args)
            else:
                self.decoder = model2decoder[args.model](args)

        # overwrite manifold for HGCN
        if self.modelname.lower() == 'hgcn':
            self.manifold = self.encoder.manifolds[-1]

        self.loss_weights = args.loss_weights
        self.n_classes = args.n_classes
        self.margin = args.margin
        if args.n_classes > 2:
            self.f1_average = 'macro'
        else:
            self.f1_average = 'binary'
        self.loss = args.loss

    def decode(self, h, edge_index, edge_attr, idx):
        # needed because the decoder is Euclidean linear
        if self.modelname.lower() == 'hgcn':
            h = self.manifold.logmap0(h)
            h = proj_tan0(self.manifold, h)
        output = self.decoder(h, edge_index, edge_attr)
        return output[idx]

    def compute_loss(self, exp, embeddings, data, split, mode='eval'):
        mask = data[f"{split}_mask"][:, exp]
        idx = mask.nonzero(as_tuple=False).view(-1)

        if self.loss_weights and self.n_classes == 2:
            weights = torch.Tensor([1.0, 1. / data.y[mask].float().mean()]).to(device=data.x.device)
        elif self.loss_weights and self.n_classes > 2:
            # Multiclass case: inverse class frequency weighting
            y_masked = data.y
            class_counts = torch.bincount(y_masked, minlength=self.n_classes).float()

            # Avoid division by zero
            class_counts[class_counts == 0] = 1.0
            weights = 1.0 / class_counts
            weights[0] = 0.0
            weights = weights / weights[1:].mean()  # optional normalization
            weights = weights.to(device=data.x.device)
        else:
            weights = torch.Tensor([1.] * self.n_classes).to(device=data.x.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        with torch.set_grad_enabled(mode == "train"):
            output = self.decode(embeddings, data.edge_index, data.edge_attr, idx)
            if self.loss == 'margin':
                correct = output.gather(1, data.y[idx].unsqueeze(-1))
                loss = F.relu(self.margin - correct + output).mean()
            elif self.loss == 'ce':
                loss = criterion(output, data.y[idx])
            else:
                raise ValueError(f"Unknown loss function: {self.loss}. Supported: 'margin', 'ce'.")
        return loss

    def compute_metrics(self, exp, embeddings, data, split):

        mask = data[f"{split}_mask"][:, exp]
        idx = mask.nonzero(as_tuple=False).view(-1)
        output = self.decode(embeddings, data.edge_index, data.edge_attr, idx)
        preds = output.argmax(dim=1)
        acc = (preds == data.y[idx]).sum().item() / idx.size(0)
        f1_score = compute_f1_score(data.y[idx].cpu().numpy(), preds.cpu().numpy(), average=self.f1_average)

        metrics = {'acc': acc, 'f1_score': f1_score}

        return metrics

    def which_metrics(self):
        return 'acc', 'f1_score'

    def init_best_metric(self, key="f1_score"):
        return {f'{key}': -1}

    def has_improved(self, m1, m2, key="f1_score"):
        if m1[key] < m2[key]:
            return True, m2
        else:
            return False, m1

    def wandb_log(self, exp, epoch, split, metrics, best_eval=False):
        if best_eval == True:
            log_data = {
                "exp": exp,
                f"best_epoch": epoch,
                f"best_{split}_acc": metrics['acc'],
                f"best_{split}_f1": metrics['f1_score']
            }
        else:
            log_data = {
                "epoch": epoch,
                f"{split}_loss": metrics['loss'],
                f"{split}_acc": metrics['acc'],
                f"{split}_f1": metrics['f1_score']
            }

        wandb.log(log_data)


class NRModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NRModel, self).__init__(args)
        args.n_classes = 1  # regression output
        if self.manifold_name in ['Lorentz', 'PoincareBall']:
            self.decoder = model2decoder[args.model](args)
        else:
            if args.model.lower() == 'hybonet':
                self.decoder = model2decoder['MLP'](args)
            else:
                self.decoder = model2decoder[args.model](args)

        # overwrite manifold for HGCN
        if self.modelname.lower() == 'hgcn':
            self.manifold = self.encoder.manifolds[-1]

        self.loss_weights = args.loss_weights
        self.n_classes = args.n_classes
        self.loss = args.loss
        if self.loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif self.loss == 'stress':
            self.criterion = StressLossNR()
        else:
            raise ValueError(f"Unknown loss function: {self.loss}. Supported: 'mse', 'stress'.")

    def decode(self, h, edge_index, edge_attr, idx):
        # needed because the decoder is Euclidean linear
        if self.modelname.lower() == 'hgcn':
            h = self.manifold.logmap0(h)
            h = proj_tan0(self.manifold, h)
        output = self.decoder(h, edge_index, edge_attr)
        return output[idx]

    def compute_loss(self, exp, embeddings, data, split, mode='eval'):
        mask = data[f"{split}_mask"][:, exp]
        idx = mask.nonzero(as_tuple=False).view(-1)
        with torch.set_grad_enabled(mode == "train"):
            output = self.decode(embeddings, data.edge_index, data.edge_attr, idx)
            loss = self.criterion(output.squeeze(), data.y[idx].float())
        return loss

    def compute_metrics(self, exp, embeddings, data, split):
        mask = data[f"{split}_mask"][:, exp]
        idx = mask.nonzero(as_tuple=False).view(-1)
        stress_crit = StressLossNR()

        with (torch.no_grad()):
            output = self.decode(embeddings, data.edge_index, data.edge_attr, idx)
            stress = stress_crit(output.squeeze(), data.y[idx].float())
            mse = F.mse_loss(output.squeeze(), data.y[idx].float())
            mae = F.l1_loss(output.squeeze(), data.y[idx].float())
            rmse = torch.sqrt(mse)

        if self.modelname.lower() == 'hgcn':
            if self.manifold.name.lower() == 'lorentz':
                curvature = 1 / self.manifold.k  # curvature magnitude value
            elif self.manifold.name.lower() == 'poincareball':
                curvature = self.manifold.c  # curvature magnitude value

            return {
                "stress": stress.item(),
                "MSE": mse.item(),
                "MAE": mae.item(),
                "RMSE": rmse.item(),
                "curvature": curvature.item()
            }
        else:
            return {
                "stress": stress.item(),
                "MSE": mse.item(),
                "MAE": mae.item(),
                "RMSE": rmse.item()
            }

    def which_metrics(self):
        if self.modelname.lower() == 'hgcn':
            return ["stress", "MSE", "MAE", "RMSE", "curvature"]
        else:
            return ["stress", "MSE", "MAE", "RMSE"]

    def init_best_metric(self, key="stress"):
        return {key: 1e9}

    def has_improved(self, m1, m2, key="stress"):
        if m1[key] > m2[key]:
            return True, m2
        else:
            return False, m1

    def wandb_log(self, exp, epoch, split, metrics, best_eval=False):
        if best_eval:
            log_data = {
                "exp": exp,
                "best_epoch": epoch,
                f"best_{split}_stress": metrics["stress"],
                f"best_{split}_MSE": metrics["MSE"],
                f"best_{split}_MAE": metrics["MAE"],
                f"best_{split}_RMSE": metrics["RMSE"],
            }
        else:
            log_data = {
                "epoch": epoch,
                f"{split}_stress": metrics["stress"],
                f"{split}_MSE": metrics["MSE"],
                f"{split}_MAE": metrics["MAE"],
                f"{split}_RMSE": metrics["RMSE"],
            }

        if self.modelname.lower() == 'hgcn':
            try:
                if self.manifold.name.lower() == 'lorentz':
                    c_last = 1 / self.manifold.k
                elif self.manifold.name.lower() == 'poincareball':
                    c_last = self.manifold.c
                if c_last is not None:
                    if best_eval:
                        c_val = c_last.item()
                        log_data["geometry/best_final_curvature"] = c_val
                    else:
                        c_val = c_last.item()
                        log_data["geometry/final_curvature"] = c_val
            except Exception as e:
                print(f"[wandb_log] Could not log curvature: {e}")

        wandb.log(log_data)


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dataset_name = args.dataset_name
        #self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        if args.loss.lower() == 'bce':
            self.loss = BCELoss(torch.nn.Sigmoid())
        elif args.loss.lower() == 'bce_fd':
            self.loss = BCELoss(FermiDiracDecoder(r=2.0, t=1.0))
        elif args.loss.lower() == 'margin':
            self.loss = MarginLoss(args.margin)
        elif args.loss.lower() == 'bce_logits':
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {args.loss}. Supported: 'bce', 'margin'.")

        # overwrite manifold for HGCN
        if self.modelname.lower() == 'hgcn':
            self.manifold = self.encoder.manifolds[-1]

    def decode(self, h, edge_label_index):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[edge_label_index[0, :], :]
        emb_out = h[edge_label_index[1, :], :]
        if self.manifold_name == 'PoincareBall':
            sqdist = self.manifold.dist2(emb_in, emb_out)
        elif self.manifold_name == 'Euclidean':
            sqdist = self.manifold.sqdist(emb_in, emb_out)
        else:
            sqdist = self.manifold.sqdist(emb_in, emb_out)
        return -sqdist

    def compute_loss(self, exp, embeddings, data, split, mode='eval'):
        if self.dataset_name in "pubmed":
            with torch.set_grad_enabled(mode == "train"):
                batch_size = 100000
                edge_index = data.edge_label_index
                labels = data.edge_label
                loss = torch.tensor(0.0, device=embeddings.device)
                #n_batches = (edge_index.size(1) + batch_size - 1 // batch_size) 

                for i in range(0, edge_index.size(1), batch_size):
                    # firt shuffle edges
                    perm = torch.randperm(edge_index.size(1))
                    edge_index = edge_index[:, perm]
                    labels = labels[perm]
                    batch_edges = edge_index[:, i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]

                    preds = self.decode(embeddings, batch_edges)
                    batch_loss = self.loss(preds, batch_labels.float())

                    loss += batch_loss * batch_edges.size(1)  # weighted sum

                # average over all edges
                loss = loss / edge_index.size(1)
            return loss
    
        else:
            with torch.set_grad_enabled(mode == "train"):
                preds = self.decode(embeddings, data.edge_label_index)
                loss = self.loss(preds, data.edge_label.float())
            return loss

    def compute_metrics(self, exp, embeddings, data, split):
        preds = self.decode(embeddings, data.edge_label_index)
        roc = roc_auc_score(data.edge_label.cpu().numpy(), preds.detach().cpu().numpy())
        ap = average_precision_score(data.edge_label.cpu().numpy(), preds.detach().cpu().numpy())

        if self.modelname.lower() == 'hgcn':
            if self.manifold.name.lower() == 'lorentz':
                curvature = 1 / self.manifold.k   # curvature magnitude value
            elif self.manifold.name.lower() == 'poincareball':
                curvature = self.manifold.c   # curvature magnitude value
            
            metrics = {'roc': roc, 'ap': ap, 'curvature': curvature}
        else:
            metrics = {'roc': roc, 'ap': ap}
        return metrics

    def which_metrics(self):
        if self.modelname.lower() == 'hgcn':
            return 'roc', 'ap', 'curvature'
        else:
            return 'roc', 'ap'

    def init_best_metric(self, key="acc"):
        return {'loss': 1e9, 'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2, key=None):
        if 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap']):
            return True, m2
        else:
            return False, m1

    def wandb_log(self, exp, epoch, split, metrics, best_eval=False):
        if best_eval == True:
            log_data = {
                "exp": exp,
                f"best_epoch": epoch,
                f"best_{split}_roc": metrics['roc'],
                f"best_{split}_ap": metrics['ap']
            }
        else:
            log_data = {
                "epoch": epoch,
                f"{split}_loss": metrics['loss'],
                f"{split}_roc": metrics['roc'],
                f"{split}_ap": metrics['ap']
            }

        if self.modelname.lower() == 'hgcn':
            try:
                if self.manifold.name.lower() == 'lorentz':
                    c_last = 1 / self.manifold.k
                elif self.manifold.name.lower() == 'poincareball':
                    c_last = self.manifold.c
                if c_last is not None:
                    if best_eval:
                        c_val = c_last.item() 
                        log_data["geometry/best_final_curvature"] = c_val
                    else:
                        c_val = c_last.item() 
                        log_data["geometry/final_curvature"] = c_val
            except Exception as e:
                print(f"[wandb_log] Could not log curvature: {e}")

        wandb.log(log_data)


class PairsModel(BaseModel):
    """
    Pair regression model (LP-style Data)
    """

    def __init__(self, args):
        super().__init__(args)
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(1.0))

        # overwrite manifold for HGCN
        if self.modelname.lower() == 'hgcn':
            self.manifold = self.encoder.manifolds[-1]

    def decode(self, h, idx):
        """
        h: node embeddings [num_nodes, d]
        idx: pair indices [2, num_pairs],  each row = [source, target]
        """

        h_src = h[idx[0]]  # idx[0]: source nodes
        h_dst = h[idx[1]]  # idx[1]: target nodes
        dist = self.manifold.dist(h_src, h_dst)
        return self.a + self.b * dist
        
    def compute_loss(self, exp, embeddings, data_split, split, mode='eval'):
        idx = data_split.pos_edge_label_index  # [2, num_pairs]
        target = data_split.pos_edge_label     # [num_pairs]

        criterion = StressLoss()
        with torch.set_grad_enabled(mode=="train"):
            output = self.decode(embeddings, idx)
            loss = criterion(output, target)
        return loss

    def compute_metrics(self, exp, embeddings, data_split, split):
        idx = data_split.pos_edge_label_index
        target = data_split.pos_edge_label
        
        criterion = StressLoss()
        with torch.no_grad():
            output = self.decode(embeddings, idx)
            stress = criterion(output, target)
            mae = F.l1_loss(output, target)
            rmse = torch.sqrt(F.mse_loss(output, target))

        if self.modelname.lower() == 'hgcn':
            if self.manifold.name.lower() == 'lorentz':
                curvature = 1 / self.manifold.k   # curvature magnitude value
            elif self.manifold.name.lower() == 'poincareball':
                curvature = self.manifold.c   # curvature magnitude value

            return {
                "stress": stress.item(),
                "MAE": mae.item(),
                "RMSE": rmse.item(),
                "curvature": curvature.item()
            }
        else:
            return {
                "stress": stress.item(),
                "MAE": mae.item(),
                "RMSE": rmse.item()
            }
    
    def which_metrics(self):
        if self.modelname.lower() == 'hgcn':
            return ["stress", "MAE", "RMSE", "curvature"]
        else:
            return ["stress", "MAE", "RMSE"]

    def init_best_metric(self, key="stress"):
        return {key: 1e9}

    def has_improved(self, m1, m2, key="stress"):
        if m1[key] > m2[key]:
            return True, m2
        else:
            return False, m1

    def wandb_log(self, exp, epoch, split, metrics, best_eval=False):
        if best_eval:
            log_data = {
                "exp": exp,
                "best_epoch": epoch,
                f"best_{split}_stress": metrics["stress"],
                f"best_{split}_MAE": metrics["MAE"],
                f"best_{split}_RMSE": metrics["RMSE"],
            }
        else:
            log_data = {
                "epoch": epoch,
                f"{split}_stress": metrics["stress"],
                f"{split}_MAE": metrics["MAE"],
                f"{split}_RMSE": metrics["RMSE"],
            }

        if self.modelname.lower() == 'hgcn':
            try:
                if self.manifold.name.lower() == 'lorentz':
                    c_last = 1 / self.manifold.k
                elif self.manifold.name.lower() == 'poincareball':
                    c_last = self.manifold.c
                if c_last is not None:
                    if best_eval:
                        c_val = c_last.item() 
                        log_data["geometry/best_final_curvature"] = c_val
                    else:
                        c_val = c_last.item() 
                        log_data["geometry/final_curvature"] = c_val
            except Exception as e:
                print(f"[wandb_log] Could not log curvature: {e}")

        wandb.log(log_data)