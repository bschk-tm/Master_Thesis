def write_network_summary_supp(model, X_test, y_test, calculation_time, best_layers=None,best_units=None,best_activation=None,best_learning_rate=None, best_batch_size=None, best_patience=None, best_dropout_rate=None, num_search_iterations=None):
    with open(f'Results/Protocols/output_{model.__class__.__name__}_supp.txt', 'w') as file:
      file.write(f""" Layer: {best_layers}
                      Units: {best_units}
                      Activation: {best_activation}
                      Learning Rate: {best_learning_rate}
                      Batch Size: {best_batch_size}
                      Patience: {best_patience}
                      Dropout Rate: {best_dropout_rate}
                      Number of Search Iterations: {num_search_iterations}
                      Model Name: {model.__class__.__name__}
                      Loss on Supported Test Dataset: {model.evaluation(X_test,y_test)[1]}
                      Absolute Error Left Boundary (x=0): {model.evaluation(X_test,y_test)[0][0]}
                      Absolute Error Right Boundary (x=L): {model.evaluation(X_test,y_test)[0][-1]}
                      Calculation Time: {calculation_time} seconds """)