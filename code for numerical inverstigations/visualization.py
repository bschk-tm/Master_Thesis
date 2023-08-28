import matplotlib.pyplot as plt
import numpy as np
import csv

def visualize(X_test, y_test, models, histories, evaluations, predictions, composite_loss, loss_terms):

    loss = []
    loss_terms_data = {}

    # save predictions into csv file    
    for i, prediction in enumerate(predictions):
        filename = f'Results/Prediction/prediction_{models[i].__class__.__name__}.csv'
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['X_test', 'Prediction'])
            for x, pred in zip(X_test, prediction.numpy()[:,0]):
                writer.writerow([x, pred])


    for history in histories:
        if composite_loss:
            loss.append(history.history['loss'])
        if loss_terms:
            for key in history.history.keys():
                if key.startswith('output_'):
                    if key not in loss_terms_data:
                        loss_terms_data[key] = []
                    loss_terms_data[key].append(history.history[key])

    

    if composite_loss:
        # Test-Train Loss Curve
        plt.figure("Train and Test Loss Curves",  figsize=(12, 6))
        plt.title("Loss Curves over Epochs")
        colors = [(0, 0.447, 0.741), (0.0, 0.5, 0.0)]
        for i, loss_values in enumerate(loss):
            plt.semilogy(histories[i].epoch, loss_values, label=f'training_loss_{models[i].__class__.__name__}', c=colors[i])
            plt.axhline(evaluations[i][1], label=f'test_loss_{models[i].__class__.__name__}', c=colors[i])
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid()
        filename = f'Results/Loss Curve/Train_Test_Losses.png'
        plt.savefig(filename, dpi=300)

        # training composite loss curve
        plt.figure("Training Loss Curve",  figsize=(12, 6))
        plt.title("Training Loss Curve over Epochs")
        colors = [(0, 0.447, 0.741), (0.0, 0.5, 0.0)]
        for i, loss_values in enumerate(loss):
            plt.semilogy(histories[i].epoch, loss_values, label=f'training_composite_loss_{models[i].__class__.__name__}', c=colors[i])
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid()
        filename = f'Results/Loss Curve/Train_Composite_Loss.png'
        plt.savefig(filename, dpi=300)

    if composite_loss and loss_terms:

        plt.figure("Training Loss Curve all Terms",  figsize=(12, 6))
        plt.title("Training Loss Curve over Epochs")
        colors = plt.cm.tab10(np.linspace(0, 1, len(models) * len(loss_terms_data.keys())))
        

        for key in loss_terms_data.keys():
            for i, loss_values in enumerate(loss):
                plt.semilogy(histories[i].epoch, loss_values, label=f'composite_loss_{models[i].__class__.__name__}')
                

            plt.semilogy(histories[i].epoch, loss_terms_data[key][i], label=f'{key}_Loss_{models[i].__class__.__name__}')
            
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.grid()
        filename = f'Results/Loss Curve/Train_All_Loss_Terms.png'
        plt.savefig(filename, dpi=300)

        # Export loss terms to CSV
        with open('Results/Loss Curve/Loss_Terms.csv', mode='w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Composite Loss']
            for key in loss_terms_data.keys():
                for i in range(len(models)):
                    fieldnames.append(f'{key}_Loss_{models[i].__class__.__name__}')
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)

            for epoch, composite in zip(histories[0].epoch, loss[0]):
                row = [epoch, composite]
                for key in loss_terms_data.keys():
                    for i in range(len(models)):
                        if i < len(loss_terms_data[key]):
                            row.append(loss_terms_data[key][i][epoch])
                        else:
                            row.append(None)
                writer.writerow(row) 

    # Numerical vs. Analytical Solution over x on Test Dataset
    y_test_min = np.min(y_test)
    y_test_max = np.max(y_test)

    plt.figure("Prediction vs. Analytical Solution",  figsize=(12, 6))
    plt.title("Prediction on Test Dataset")
    for i, prediction in enumerate(predictions):
        plt.scatter(X_test, prediction, label=f'{models[i].__class__.__name__}', color=colors[i], marker='x', s=20)
    plt.plot(X_test, y_test, label='Analytical solution', color='orange')
    plt.xlabel("Spatial Coordinate x [m]")
    plt.ylabel("Bending w(x) [m]")
    plt.ylim(y_test_min, y_test_max*1.1)
    plt.legend()
    plt.grid()
    filename = f'Results/Prediction/prediction_{models[i].__class__.__name__}.png'
    plt.savefig(filename, dpi=300)  

    plt.tight_layout()
    plt.show()
