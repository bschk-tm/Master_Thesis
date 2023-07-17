import matplotlib.pyplot as plt

def visualize(history,model, evaluation, x_test,prediction, y_test, composite_loss: bool,val_loss: bool, loss_terms: bool):

  # extracting relevant Loss Terms
  if composite_loss and val_loss:  
    loss = history.history['loss']
    validation_loss = history.history['val_loss']
  elif composite_loss:
    loss = history.history['loss']

    

  if loss_terms:
    pde_loss = history.history['output_1_loss']
    bc_w_loss = history.history['output_2_loss']
    bc_w_x_loss = history.history['output_3_loss']
    bc_M_loss = history.history['output_4_loss']
    bc_V_loss = history.history['output_5_loss']

  plt.figure("Results")

  # Training Loss Curve
  plt.subplot(211)
  plt.title("Loss Curves over Epochs")
  if composite_loss and val_loss: 
    plt.semilogy(history.epoch, loss, label='training_loss_composite',marker='o')
    plt.semilogy(history.epoch, validation_loss, label='validation_loss_composite', c='green', marker='o')
  #plt.axhline( evaluation[1], label='Test Loss', c='green')
  elif composite_loss:
    plt.semilogy(history.epoch, loss, label='Composite Loss')



  if loss_terms:
    plt.semilogy(history.epoch, pde_loss, label='PDE Loss')
    plt.semilogy(history.epoch, bc_w_loss, label='w(0) = 0 Loss')
    plt.semilogy(history.epoch, bc_w_x_loss, label='w_x(0) = 0 Loss')
    plt.semilogy(history.epoch, bc_M_loss, label='M(L) = 0 Loss')
    plt.semilogy(history.epoch, bc_V_loss, label='V(L) = 0 Loss')

  plt.ylabel("Loss")
  plt.xlim(0,max(history.epoch))
  plt.xlabel("Epochs")
  plt.legend()
  plt.grid()

  # Numerical vs. Analytical Solution over x on Test Dataset
  ax2 = plt.subplot(212)
  ax2.plot(x_test, prediction, label=f'{model.__class__.__name__}')
  ax2.plot(x_test, y_test, label='Analytical solution')
  ax2.set_title("Evaluation on Test Dataset")
  ax2.set_xlabel("Spatial Coordinate x [m]")
  ax2.set_ylabel("Bending w(x) [m]")
  ax2.set_ylim(-0.13,0.025)
  ax2.legend()
  ax2.grid()


  # Absolute Error over x on Test Dataset
  plt.figure("Absolute Error on Test Dataset")
  plt.ylim(0,max(evaluation[0]))
  plt.title("Absolute Error |w_PINN - w_Analytical| on Test Dataset")
  plt.xlabel("Spatial Coordinate x [m]")
  plt.ylabel("Absolute Error [m]")
  plt.plot(x_test.numpy(), evaluation[0], c='red',label='Absolute Error PINN vs. Analytical')
  plt.legend()
  plt.grid()

  plt.tight_layout()
  plt.show()
