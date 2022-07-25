import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import get_logger
from utils.test_env import EnvTest
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear


from configs.q4_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            (stride - 1) * img_height - stride + filter_size) // 2

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n

        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################


        print( "model init, original state shape:", state_shape, " state_history:", self.config.state_history )

        channels = n_channels * self.config.state_history

        output_size = img_height  # image size

        out_channle_1 = 32  # filters
        filter_size_1 = 8
        strider_1 = 4 
        padding_1 = ((strider_1 - 1) * output_size - strider_1 + filter_size_1) // 2
        output_size = ( output_size + 2*padding_1 - filter_size_1 ) // strider_1 + 1
        # print( f"padding: {padding_1}, output_size: {output_size}" )

        out_channle_2 = 64  # filters
        filter_size_2 = 4
        strider_2 = 2
        padding_2 = ((strider_2 - 1) * output_size - strider_2 + filter_size_2) // 2
        output_size = (output_size + 2*padding_2 - filter_size_2) // strider_2 + 1
        # print( f"padding: {padding_2}, output_size: {output_size}" )

        out_channel_3 = 64  # filters
        filter_size_3 = 3
        strider_3 = 1
        padding_3 = ((strider_3 - 1) * output_size - strider_3 + filter_size_3) // 2
        output_size = (output_size + 2*padding_3 - filter_size_3) // strider_3 + 1
        # print( f"padding: {padding_3}, output_size: {output_size}" )

        # NOTE: nn.Conv2d only support HCHW format

        for network in ["q_network", "target_network"]:
            model = nn.Sequential(
                nn.Conv2d(      channels ,out_channle_1, filter_size_1, stride=strider_1, padding=padding_1 ),
                nn.ReLU(),
                nn.Conv2d( out_channle_1 ,out_channle_2, filter_size_2, stride=strider_2, padding=padding_2 ),
                nn.ReLU(),
                nn.Conv2d( out_channle_2 ,out_channel_3, filter_size_3, stride=strider_3, padding=padding_3 ),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear( out_channel_3 * img_width * img_height , 512 ),
                nn.ReLU(),
                nn.Linear( 512 , num_actions ),
            )
            setattr(self, network, model )

        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None

        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################

        nn_net = getattr(self, network)
        # nn.Conv2d want the input in format( N,C,H,W ) , but we have ( N,H,W,C )
        # NHWC -> NCHW
        _state = torch.permute(state, ( 0,3,1,2 ) )
        # print( "adjusted state_shape:", _state.shape )
        out = nn_net( _state ) 
        # print("model outshape:", out.shape)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
