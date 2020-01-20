import os,sys 
import logging
import argparse
import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from model import LSTM_model



def parser():
    parser = argparse.ArgumentParser(description='Mnist LSTM')
    parser.add_argument('--hidden_size', type=int, default=128, help='Which model to use')
    parser.add_argument('--data_root', default='data',  help='the directory to save the dataset')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    

    ## Training realted 
    parser.add_argument('--batch_size', '-b', type=int, default=100, help='batch size')
    parser.add_argument('--max_epoch', '-m_e', type=int, default=20, 
        help='the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_eval_step', type=int, default=100, 
        help='number of iteration per one evaluation')
    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    parser.add_argument('--seed', default=1111, help='The random seed')
    parser.add_argument('--alpha', type=float, default=0.0001, help="Test")
    parser.add_argument('--mask', action='store_true', help='whether to use masked model')


    return parser.parse_args()

def main(args):
    
    sequence_length = 28 
    input_size = 28
    num_layers = 2
    num_classes = 10
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_epochs = args.max_epoch
    mask = args.mask
    alpha = args.alpha

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = args.log_root
    data_dir = args.data_root

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    ###############################################################################
    # Logging
    ###############################################################################
    model_config_name = 'nhid:{}-nlayer:{}-epoch:{}-mask:{}-alpha:{}'.format(
        hidden_size, num_layers, num_epochs, mask, alpha
    )

    log_file_name = "{}.log".format(model_config_name)
    log_file_path = os.path.join(log_dir, log_file_name)
    logging.basicConfig(filename=log_file_path, 
                        level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        )
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))


    ###############################################################################
    # Loading data
    ###############################################################################

    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    model = LSTM_model(input_size, hidden_size, num_layers, num_classes, mask=mask).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):

        hidden = model.init_hidden(batch_size)

        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            hidden = repackage_hidden(hidden)
            # Forward pass
            outputs, hidden = model(images, hidden)
            loss = criterion(outputs, labels)
            sparse_regularization = torch.tensor(0.).to(device)
            for name, param in model.named_parameters():
                if name.find('threshold') != -1:
                    sparse_regularization += torch.sum(torch.exp(-param))
            loss = loss + alpha * sparse_regularization
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                logger.info('|Epoch [{}/{}] | Step [{}/{}] | Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                if mask:
                    logger.info("Lstm 1, keep ratio {:.4f}".format(model.lstm1.cell.keep_ratio))
                    logger.info("Lstm 2, keep ratio {:.4f}".format(model.lstm2.cell.keep_ratio))
                    logger.info("Model keep ratio {:.4f}".format(model.keep_ratio))

    logger.info("Training process finish")

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0

        hidden = model.init_hidden(batch_size)
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(images, hidden)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        logger.info('Test Accuracy of the model on the 10000 test images: {:.5f}'.format(100.0 * correct / total)) 

if __name__ == '__main__':
    args = parser()
    #print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)