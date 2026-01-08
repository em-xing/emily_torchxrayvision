import os
import pickle
import pprint
import random
from glob import glob
import pdb
from os.path import exists, join

import numpy as np
import torch
import sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import sklearn, sklearn.model_selection
import torchxrayvision as xrv
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm as tqdm_base
def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)
#from tqdm.auto import tqdm




def train(model, dataset, cfg, valid_dataset=None, use_softmax=False):
    print("Our config:")
    pprint.pprint(cfg)
        
    dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
    
    device = 'cuda' if cfg.cuda else 'cpu'
    if not torch.cuda.is_available() and cfg.cuda:
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')

    print(cfg.output_dir)

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    
    # Setting the seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Dataset
    if valid_dataset is None:
        # give patientid if not exist
        if "patientId" not in dataset.csv.columns:
            dataset.csv["patientId"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
            
        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8,test_size=0.2, random_state=cfg.seed)
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientId))
        train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
        valid_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)
    else:
        train_dataset = dataset

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)
    #print(model)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
    print(optim)
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[20, 40], gamma=0.2)

    if use_softmax:
        criterion = 'softmax'
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validauc = None
    auc_test = None
    metrics = []
    weights_files = glob(join(cfg.output_dir, f'{dataset_name}-e*.pt'))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(cfg.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        model.load_state_dict(torch.load(weights_file).state_dict())

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        best_metric = metrics[-1]['best_metric']
        weights_for_best_validauc = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)
    
    for epoch in range(start_epoch, cfg.num_epochs):

        avg_loss = train_epoch(cfg=cfg,
                               epoch=epoch,
                               model=model,
                               device=device,
                               optimizer=optim,
                               train_loader=train_loader,
                               criterion=criterion)
        
        auc_valid, task_aucs_valid, task_outputs_valid, task_targets_valid = valid_test_epoch(name='Valid',
                                     epoch=epoch,
                                     model=model,
                                     device=device,
                                     data_loader=valid_loader,
                                     criterion=criterion)

        if cfg.visualize_spline:
            # Check if model has windowing functionality
            windowing_function = None
            
            # Check direct windowing function
            if hasattr(model, 'windowing_function'):
                windowing_function = model.windowing_function
            elif hasattr(model, 'module') and hasattr(model.module, 'windowing_function'):
                # Handle DataParallel case
                windowing_function = model.module.windowing_function
            
            if windowing_function and hasattr(windowing_function, 'visualize_mapping'):
                try:
                    # Get the mapping for visualization
                    input_vals, output_vals = windowing_function.visualize_mapping(n_points=500)
                    
                    # Plot
                    plt.figure(figsize=(10, 6))
                    plt.plot(input_vals, output_vals, 'b-', linewidth=2, label='Learned Windowing Function')
                    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Identity (no windowing)')
                    plt.title(f'Epoch {epoch+1} - Windowing Function')
                    plt.xlabel('Input Intensity (normalized)')
                    plt.ylabel('Output Intensity (normalized)')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)
                    
                    # Save the plot
                    plot_path = join(cfg.output_dir, f'{dataset_name}-windowing_epoch_{epoch+1}.png')
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Saved windowing visualization to {plot_path}")
                except Exception as e:
                    print(f"Could not visualize windowing function: {e}")
            elif cfg.visualize_spline:
                print("No windowing function found for visualization")

        if auc_valid > best_metric:
            best_metric = auc_valid
            weights_for_best_validauc = model.state_dict()
            torch.save(model, join(cfg.output_dir, f'{dataset_name}-best.pt'))
            
            # Save detailed best checkpoint with metrics
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'best_metric': best_metric,
                'train_loss': avg_loss,
                'valid_auc': auc_valid,
                'valid_task_aucs': task_aucs_valid,
                'config': cfg
            }
            
            # Add windowing parameters if available
            if hasattr(model, 'get_windowing_params'):
                best_checkpoint['windowing_params'] = model.get_windowing_params()
            elif hasattr(model, 'module') and hasattr(model.module, 'get_windowing_params'):
                # Handle DataParallel case
                best_checkpoint['windowing_params'] = model.module.get_windowing_params()
            torch.save(best_checkpoint, join(cfg.output_dir, f'{dataset_name}-best_checkpoint.pt'))
            print(f"Saved new best model with AUC: {auc_valid:.4f}")

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validauc": auc_valid,
            'best_metric': best_metric,
            'individual_auc': dict(zip(dataset.pathologies if hasattr(dataset, 'pathologies') else range(len(task_aucs_valid)), task_aucs_valid)),
            'mean_auc': auc_valid,
            'config_snapshot': {
                'window_nbins': getattr(cfg, 'window_nbins', None),
                'batch_size': cfg.batch_size,
                'lr': cfg.lr,
                'model': cfg.model
            }
        }

        metrics.append(stat)

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
            
        # Save metrics as JSON for easier reading
        import json
        
        def convert_numpy_types(obj):
            """Recursively convert numpy types to JSON-serializable types."""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(val) for key, val in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):  # handle numpy scalars
                return obj.item()
            else:
                return obj
        
        metrics_json = []
        for m in metrics:
            m_copy = convert_numpy_types(m)
            metrics_json.append(m_copy)
        
        with open(join(cfg.output_dir, f'{dataset_name}-metrics.json'), 'w') as f:
            json.dump(metrics_json, f, indent=2)

        # Save epoch checkpoint every 5 epochs or if windowing is enabled
        if (epoch + 1) % 5 == 0 or getattr(cfg, 'window_nbins', None) is not None:
            epoch_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'train_loss': avg_loss,
                'valid_auc': auc_valid,
                'valid_task_aucs': task_aucs_valid,
                'best_metric_so_far': best_metric,
                'config': cfg
            }
            
            # Add windowing parameters if available
            if hasattr(model, 'get_windowing_params'):
                epoch_checkpoint['windowing_params'] = model.get_windowing_params()
            elif hasattr(model, 'module') and hasattr(model.module, 'get_windowing_params'):
                # Handle DataParallel case
                epoch_checkpoint['windowing_params'] = model.module.get_windowing_params()
                
            torch.save(epoch_checkpoint, join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}_checkpoint.pt'))
        
        torch.save(model, join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}.pt'))

        if cfg.use_scheduler:
            scheduler.step()

    return metrics, best_metric, weights_for_best_validauc





def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, limit=None):
    model.train()

    if cfg.taskweights:
        weights = np.nansum(train_loader.dataset.labels, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights/weights.max()
        weights = torch.from_numpy(weights).to(device).float()
        print("task weights", weights)
    
    avg_loss = []
    t = tqdm(train_loader)
    for batch_idx, samples in enumerate(t):
        
        if limit and (batch_idx > limit):
            print("breaking out")
            break
            
        optimizer.zero_grad()
        
        images = samples["img"].float().to(device)
        targets = samples["lab"].to(device)

        outputs = model(images)

        loss = torch.zeros(1).to(device).float()
        if criterion == 'softmax':
            loss = F.cross_entropy(outputs, targets)
        else:
            for task in range(targets.shape[1]):
                task_output = outputs[:,task]
                task_target = targets[:,task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if len(task_target) > 0:
                    task_loss = criterion(task_output.float(), task_target.float())
                    if cfg.taskweights:
                        loss += weights[task]*task_loss
                    else:
                        loss += task_loss
        
        # here regularize the weight matrix when label_concat is used
        if cfg.label_concat_reg:
            if not cfg.label_concat:
                raise Exception("cfg.label_concat must be true")
            weight = model.classifier.weight
            num_labels = len(xrv.datasets.default_pathologies)
            num_datasets = weight.shape[0]//num_labels
            weight_stacked = weight.reshape(num_datasets,num_labels,-1)
            label_concat_reg_lambda = torch.tensor(0.1).to(device).float()
            for task in range(num_labels):
                dists = torch.pdist(weight_stacked[:,task], p=2).mean()
                loss += label_concat_reg_lambda*dists
                
        loss = loss.sum()
        
        if cfg.featurereg:
            feat = model.features(images)
            loss += feat.abs().sum()
            
        if cfg.weightreg:
            loss += model.classifier.weight.abs().sum()
        
        loss.backward()

        avg_loss.append(loss.detach().cpu().numpy())
        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        optimizer.step()

    return np.mean(avg_loss)

def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None):
    model.eval()

    avg_loss = []
    task_outputs={}
    task_targets={}
    n_tasks = len(np.unique(data_loader.dataset.labels)) if criterion == 'softmax' else data_loader.dataset[0]["lab"].shape[0]
    for task in range(n_tasks):
        task_outputs[task] = []
        task_targets[task] = []
        
    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            if limit and (batch_idx > limit):
                print("breaking out")
                break
            
            images = samples["img"].to(device)
            targets = samples["lab"].to(device)

            outputs = model(images)
            if criterion == 'softmax':
                outputs_softmax = F.softmax(outputs, dim=-1)
            
            loss = torch.zeros(1).to(device).double()
            for task in range(n_tasks):

                if criterion == 'softmax':
                    task_target = targets == task
                    task_output = outputs_softmax[:, task]
                else:
                    task_target = targets[:,task]
                    task_output = outputs[:, task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                if criterion != 'softmax':
                    if len(task_target) > 0:
                        loss += criterion(task_output.double(), task_target.double())
                
                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            if criterion == 'softmax':
                loss = F.cross_entropy(outputs, targets)

            loss = loss.sum()
            
            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')
            
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
    
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)])
    print(f'Epoch {epoch + 1} - {name} - Avg AUC = {auc:4.4f}')

    return auc, task_aucs, task_outputs, task_targets
