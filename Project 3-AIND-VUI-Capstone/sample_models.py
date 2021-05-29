from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None,input_dim[1]))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, input_shape = (), return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        if i == 0:
            simp_rnn = GRU(units, activation='relu',
                           return_sequences=True, implementation=2, name='rnn_'+str(i))(input_data)
        else:
            simp_rnn = GRU(units, activation='relu',
                           return_sequences=True, implementation=2, name='rnn_'+str(i))(bn_rnn)
        bn_rnn = BatchNormalization(name='bn_rnn_'+str(i))(simp_rnn) 
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn =  Bidirectional(GRU(units, return_sequences=True,
                                   implementation=2, name='rnn'), 
                               merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(units=output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, reccur_units,
                # CNN parameters
                filters=200, kernel_size=11, conv_stride=2, conv_border_mode='same', dilation=1, cnn_activation='relu',
                # RNN parameters
                recur_Bidirectional = True,
                recur_layers=2,
                recur_type='GRU',
                reccur_droput=0.2,
                recurrent_dropout=0.2,
                reccur_merge_mode='concat',
                fc_activation='relu',
                output_dim=29):
    """ Build a deep network for speech  
    """
    
    # Checks literal parameters values
    assert cnn_activation in {'relu', 'leakyrelu'} 
    assert recur_type in {'GRU', 'LSTM'}
    assert reccur_merge_mode in {'sum', 'mul', 'concat', 'ave' }

    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    nn=input_data
    
    # Add convolutional layers
    nn = Conv1D(filters,
                kernel_size,
                strides=conv_stride,
                padding=conv_border_mode,
                dilation_rate=dilation,
                activation=cnn_activation,
                name='cnn_'+str(1))(nn)

    # Add (in order) Batch Normalization,Dropout and Activation
    nn = BatchNormalization(name='bn_cnn_'+str(1))(nn)
    
    # TODO: Add bidirectional recurrent layers
    for i in range(recur_layers):
        layer_name='rnn_'+str(i)
        if recur_Bidirectional:
            if  recur_type=='GRU':
                nn =  Bidirectional(GRU(reccur_units, return_sequences=True,
                                        implementation=2,
                                        name=layer_name,
                                        dropout=reccur_droput,
                                        recurrent_dropout=recurrent_dropout),
                                    merge_mode=reccur_merge_mode)(nn)
            else:
                nn =  Bidirectional(LSTM(reccur_units, return_sequences=True,
                                         implementation=2,
                                         name=layer_name,
                                         dropout=reccur_droput,
                                         recurrent_dropout=recurrent_dropout),
                                    merge_mode=reccur_merge_mode)(nn)
        else:
            if  recur_type=='GRU':
                nn =  GRU(reccur_units, return_sequences=True,
                                        implementation=2,
                                        name=layer_name,
                                        dropout=reccur_droput,
                                        recurrent_dropout=recurrent_dropout)(nn)
            else:
                nn =  LSTM(reccur_units, return_sequences=True,
                                         implementation=2,
                                         name=layer_name,
                                         dropout=reccur_droput,
                                         recurrent_dropout=recurrent_dropout)(nn)
            
        nn = BatchNormalization(name='bn_'+layer_name)(nn) 
        

    nn = TimeDistributed(Dense(units=output_dim, name='fc_'+str(1)))(nn)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(nn)
    
    # TODO: Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    
    # TODO: Specify model.output_length: select custom or Udacity version
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
      
    print(model.summary(line_length=110))
    return model
