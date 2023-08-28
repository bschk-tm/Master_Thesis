import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint


class LossCurveCallback(ModelCheckpoint):
    def __init__(self, filepath, period, X_test, y_test, model):
        super(LossCurveCallback, self).__init__(filepath, save_weights_only=True, save_best_only=False)
        self.period = period
        self.losses = []
        self.epochs = []
        self.model = model
        self.y_test = y_test
        self.X_test = X_test
        self.class_name = model.__class__.__name__


    def on_epoch_end(self, epoch, logs=None):
        super(LossCurveCallback, self).on_epoch_end(epoch, logs)
        self.losses.append(logs['loss'])
        self.epochs.append(epoch+1) # starts counting at 0
        if (epoch+1) % self.period == 0:
            self.plot_loss_curve(epoch)
            #self.plot_abs_error_curve(epoch)
            self.plot_pinn_vs_analytical(epoch)

    def plot_loss_curve(self,epoch):
        plt.figure()
        plt.xlim(0, max(self.epochs)) 
       
        plt.semilogy(self.epochs, self.losses,label='Composite Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid()
        filename = f'Results/Loss Curve/{self.class_name}_loss_epoch_{epoch+1}.png'
        plt.savefig(filename)  
        plt.close()

    # def plot_abs_error_curve(self, epoch): # regarding test dataset

    #     absolute_errors = []

    #     pred = self.model.predict(self.X_test)
    #     absolute_error = np.abs(pred - self.y_test)
    #     absolute_errors.append(absolute_error)

    #     # Absolute Error over x on Test Dataset
    #     plt.figure("Absolute Error between Prediction and Analytical Solution on Test Dataset")
    #     for i, absolute_error in enumerate(absolute_errors):
    #         plt.plot(self.X_test, absolute_error, label=f'Absolute Error {self.model.__class__.__name__}')
    #     plt.title("Absolute Error |w_PINN - w_Analytical| on Test Dataset")
    #     plt.xlabel("Spatial Coordinate x [m]")
    #     plt.ylabel("Absolute Error [m]")
    #     plt.legend()
    #     plt.grid()
    #     filename = f'Results/Absolute Error/{self.class_name}_abs_error_epoch_{epoch+1}.png'
    #     plt.savefig(filename)  
    #     plt.close()


    def plot_pinn_vs_analytical(self, epoch):  # regarding test dataset
        prediction = self.model.predict(self.X_test)
        plt.figure()
        plt.plot(self.X_test, prediction, label='PINN solution')
        plt.plot(self.X_test, self.y_test, label='Analytical solution')
        plt.title(f"PINN vs. Analytical w(x) at Epoch {epoch+1}")
        plt.xlabel("Spatial Coordinate x [m]")
        plt.ylabel("Bending w(x) [m]")
        plt.ylim(-0.13,0.025)
        plt.legend(loc='upper right')
        plt.grid()
        filename = f'Results/Prediction/{self.class_name}_prediction_epoch_{epoch+1}.png'
        plt.savefig(filename)  
        plt.close()