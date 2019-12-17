import numpy as np

class MLP:
    def __init__(self):
        self.layers = []
        self.mode = 'TRAIN'
        
    def add(self, layer):
        '''
        add a new layer to the layers of model.
        '''
        self.layers.append(layer)
    
    def set_mode(self, mode):
        if mode == 'TRAIN' or mode == 'TEST':
            self.mode = mode
        else:
            raise ValueError('Invalid Mode')
    
    def forward(self, x, y):
        loss, scores = None, None
        ############################################################################
        # TODO: Implement the forward pass of MLP model.                           #
        # Note that you have the layers of the model in self.layers and each one   #
        # of them has a forward() method.                                          #
        # The last layer is always a LossLayer which in this assignment is only    #
        # SoftmaxCrossEntropy.                                                     #
        # You have to compute scores (output of model before applying              #
        # SoftmaxCrossEntropy) and loss which is the output of SoftmaxCrossEntropy #
        # forward pass.                                                            #
        # Do not forget to pass mode=self.mode to forward pass of the layers! It   #
        # will be used later in Dropout and Batch Normalization.                   #
        # Do not forget to add the L2-regularization loss term to loss. You can    #
        # find whether a layer has regularization_strength by using get_reg()      #
        # method. Note that L2-regularization is only used for weights of fully    #
        # connected layers in this assignment.                                     #
        ############################################################################

        loss=0
        tensor = self.layers[0].forward(x, mode=self.mode)
        if self.layers[0].get_reg :
            ww = self.layers[0].get_params()['w'].data
            loss+=self.layers[0].get_reg()*np.sum(ww*ww)

        for i in range(1, len(self.layers) - 1):
            tensor = self.layers[i].forward(tensor,mode=self.mode)
            if self.layers[i].get_reg():
                ww = self.layers[i].get_params()['w'].data
                loss += self.layers[i].get_reg() * np.sum(ww * ww)


        loss += self.layers[-1].forward(tensor, y)
        scores=tensor

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return scores, loss
        
        
    def backward(self):
        ############################################################################
        # TODO: Implement the backward pass of the model. Use the backpropagation  #
        # algorithm.                                                               #                         
        # Note that each one of the layers has a backward() method and the last    #
        # layer would always be a SoftmaxCrossEntropy layer.                       #
        ############################################################################
        dout=self.layers[-1].backward()
        for i in range(len(self.layers)-2,-1,-1):
            dout=self.layers[i].backward(dout)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
            
    def __str__(self):
        '''
        returns a nice representation of model
        '''
        splitter = '===================================================='
        return splitter + '\n' + '\n'.join('layer_{}: '.format(i) + 
                                           layer.__str__() for i, layer in enumerate(self.layers)) + '\n' + splitter
