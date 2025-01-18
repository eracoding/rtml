import time
import copy

import torch
import matplotlib.pyplot as plt


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25, weights_name='weight_save', is_inception=False):
    before = time.time()

    val_acc_history = []
    loss_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-"*10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for imgs, labels in dataloader[phase]:
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    if is_inception and phase == "train":
                        outputs, aux1, aux2 = model(imgs, train=True)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux1, labels)
                        loss3 = criterion(aux2, labels)
                        loss = loss1 + 0.3 * loss2 + 0.3 * loss3
                    else:
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1) # _ - max values, and preds - indices -> classes
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)
            
            elapsed_epoch = time.time() - epoch_start

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print("Epoch time taken: ", elapsed_epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weights_name + '.pth')
            
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
            
            if phase == 'train':
                loss_acc_history.append(epoch_loss)
        
        print()

    time_elapsed = time.time() - before
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, loss_acc_history

def plot_data(val_acc_history, loss_acc_history, model_name="alexnet"):
    plt.plot(loss_acc_history, label = 'Validation')
    plt.title('Loss per epoch')
    plt.legend()
    plt.savefig(f"{model_name}_val_acc_history.png")
    plt.plot(val_acc_history, label = 'Validation')
    plt.title('Accuracy per epoch')
    plt.legend()
    plt.savefig(f"{model_name}_loss_acc_history.png")
