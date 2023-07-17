import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


class LossCurveCallback(ModelCheckpoint):
    def __init__(self, filepath, period, x_test, y_test, model):
        super(LossCurveCallback, self).__init__(filepath, save_weights_only=True, save_best_only=False)
        self.period = period
        self.losses = []
        self.epochs = []
        self.model = model
        self.y_test = y_test
        self.x_test = x_test
        self.class_name = model.__class__.__name__


    def on_epoch_end(self, epoch, logs=None):
        super(LossCurveCallback, self).on_epoch_end(epoch, logs)
        self.losses.append(logs['loss'])
        self.epochs.append(epoch+1) # starts counting at 0
        if (epoch+1) % self.period == 0:
            self.plot_loss_curve(epoch)
            self.plot_abs_error_curve(epoch)
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
        filename = f'Loss Curve/{self.class_name}_loss_curve_epoch_{epoch+1}.png'
        plt.savefig(filename)  
        plt.close()

    def plot_abs_error_curve(self, epoch): # regarding test dataset

        abs_error = self.model.evaluation(x_test=self.x_test, y_test=self.y_test)[0]

        plt.figure()
        plt.ylim(0, 0.005) 
        plt.plot(self.x_test, abs_error,'r', label='Absolute Error')
        plt.title(f'Absolute Error |w_PINN - w_Analytical| on Test Dataset at Epoch {epoch+1}')
        plt.xlabel('Spatial Coordinate x [m]')
        plt.ylabel('Absolute Error [m]')
        plt.legend(loc='upper right')
        plt.grid()
        filename = f'Absolute Error/{self.class_name}_abs_error_curve_epoch_{epoch+1}.png'
        plt.savefig(filename)  
        plt.close()

    def plot_pinn_vs_analytical(self, epoch):  # regarding test dataset
        prediction = self.model.predict(self.x_test)
        plt.figure()
        plt.plot(self.x_test, prediction, label='PINN solution')
        plt.plot(self.x_test, self.y_test, label='Analytical solution')
        plt.title(f"PINN vs. Analytical w(x) at Epoch {epoch+1}")
        plt.xlabel("Spatial Coordinate x [m]")
        plt.ylabel("Bending w(x) [m]")
        plt.ylim(-0.13,0.025)
        plt.legend(loc='upper right')
        plt.grid()
        filename = f'Numerical vs Analytical/{self.class_name}_num_vs_analytical_curve_epoch_{epoch+1}.png'
        plt.savefig(filename)  
        plt.close()