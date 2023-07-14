import matplotlib.pyplot as plt

def visualize(history, evaluation, prediction,coords, y_analytical, composite_loss: bool, loss_terms: bool):


  # extracting relevant Loss Terms
  if composite_loss:  
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
  if composite_loss: 
    plt.semilogy(history.epoch, loss, label='Composite Loss')
  #plt.axhline( evaluation[1], label='Test Loss', c='green')

  if loss_terms:
    plt.semilogy(history.epoch, pde_loss, label='PDE Loss')
    plt.semilogy(history.epoch, bc_w_loss, label='w(0) = 0 Loss')
    plt.semilogy(history.epoch, bc_w_x_loss, label='w_x(0) = 0 Loss')
    plt.semilogy(history.epoch, bc_M_loss, label='M(L) = 0 Loss')
    plt.semilogy(history.epoch, bc_V_loss, label='V(L) = 0 Loss')

  plt.ylabel("Loss")
  plt.xlabel("Epochs")
  plt.legend()
  plt.grid()

  ax1 = plt.subplot(212)
  plt.title("Numerical vs. Analytical w(x)")
  lns1 = ax1.plot(coords, prediction, label='PINN solution')
  lns2 = ax1.plot(coords, y_analytical, label='Analytical solution')
  ax1.set_xlabel("Spatial Coordinate x [m]")
  ax1.set_ylabel("Bending w(x) [m]")
  ax1.set_ylim(-0.13,0.025)
  ax1.legend()
  ax1.grid()

  # Select how many 
  # step = 250
  # selected_coords = coords.numpy()[::step]
  # selected_abs_error = evaluation[0][::step]

  # ax2 = ax1.twinx()
  # ax2.set_ylabel("Absolute Error [m]")
  # lns3 = ax2.plot(selected_coords, selected_eval, c='red',label='Absolute Error PINN vs. Analytical')
  # lns = lns1+lns2+lns3
  # labs = [l.get_label() for l in lns]
  # ax1.legend(lns, labs, loc='center left')

  # plt.figure(2)
  # plt.title("Absolute Error of w(x) between PINN and Analytical Solution")
  # # plt.plot(selected_coords, selected_abs_error, c='red',label='Absolute Error PINN vs. Analytical')
  # plt.xlabel("Spatial Coordinate x [m]")
  # plt.ylabel("Absolute Error [m]")
  # plt.plot(coords.numpy(), evaluation[0], c='red',label='Absolute Error PINN vs. Analytical')
  # plt.legend()

  plt.tight_layout()
  plt.show()
