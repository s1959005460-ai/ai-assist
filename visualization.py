import matplotlib.pyplot as plt

def plot_loss(loss_list):
    plt.figure()
    for client_id, losses in enumerate(loss_list):
        plt.plot(losses, label=f"Client {client_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
