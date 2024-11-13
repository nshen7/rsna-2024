# # ONLY NEED TO RUN ONCE
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py")
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py")
import os
import utils
from engine import train_one_epoch, evaluate

# Util functions
with open(os.path.join(SRC_DIR, 'data.py')) as file:
    exec(file.read())
with open(os.path.join(SRC_DIR, 'models.py')) as file:
    exec(file.read())
    
    
# LEVEL_LABELS = {
#     "L1/L2": 1,
#     "L2/L3": 2,
#     "L3/L4": 3,
#     "L4/L5": 4,
#     "L5/S1": 5
# }

def model_pipeline(config, model, model_dir, train_df, val_df):

    # make the model, data, and optimization problem
    model, train_loader, val_loader, optimizer, lr_scheduler = make(config, model, train_df, val_df)

    # and use them to train the model
    train_and_validate(model, model_dir, train_loader, val_loader, optimizer, lr_scheduler, config)

    return


def make(config, model, train_df, val_df):
    
    # Make training set
    dataset = RSNAMultipleBBoxesDataset(train_df, w = config['box_w'], h_l1_l4 = config['box_h_l1_l4'], h_l5 = config['box_h_l5'])
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=os.cpu_count()
    )
    
    # Make validation set
    dataset_val = RSNAMultipleBBoxesDataset(val_df, w = config['box_w'], h_l1_l4 = config['box_h_l1_l4'], h_l5 = config['box_h_l5'])
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=os.cpu_count()
    )

    # Make model
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=config['lr'],
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )
    
    return model, train_loader, val_loader, optimizer, lr_scheduler

def train_and_validate(model, model_dir, train_loader, val_loader, optimizer, lr_scheduler, config):
    
    for epoch in tqdm.tqdm(range(config['num_epochs']), desc="Training Epochs"):
        
        # train for one epoch, printing every 30 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=30)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the validation dataset
        evaluate(model, val_loader, device=device)
        
        # Save model after every epoch
        dirname = f'{model_dir}/epoch_{epoch}'
        os.makedirs(dirname, exist_ok=True,)
        fname = f'{dirname}/model_dict.pt'
        torch.save(model.state_dict(), fname)