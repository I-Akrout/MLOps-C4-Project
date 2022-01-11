import pytest
import logging
import starter.ml.clean_data_helper as cdh
import starter.ml.model as model
logging.basicConfig(
    filename='./ml/logs/model_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")

@pytest.mark.parametrize("load_data",[cdh.load_data])
def test_load_data_exists(load_data):
    """
    This function test the load_data helper function.
    
    The test consists of providing an existing path and expecting 
    a data frame in return. 
    """
    data = [
        'c1,c2,c3,c4',
        'First,entry,in,file',
        'Second,entry,in,file'
    ]
    data_shape = (2,4)
    
    
    logging.info('Starting the test 1 of load_data function ')

    from pathlib import Path
    from os.path import join
    from os import remove
    file_path = Path(__file__)
    file_path = join(file_path.parent,'test_file.csv')
    with open(file_path,'w') as fp:
        for line in data:
            fp.write(line+'\n')
    df = None
    try:
        df = load_data(file_path)
        remove(file_path)
    except FileNotFoundError as err:
        logging.error('ERROR: FileNotFoundError is not \
            expected in this test.')
        remove(file_path)
        raise err
    
    if df is not None:
        try:
            assert df.shape == data_shape
        except AssertionError as err :
            logging.error('ERROR: Error in loading shape')
            raise err
    
    logging.info('SUCCESS: Test ended successfully')



@pytest.mark.parametrize("load_data",[cdh.load_data])
def test_load_data_missing(load_data):
    """
    This function test the load_data helper function.
    
    The test consists of providing an unexisting path and expecting 
    an error to be raised . 
    """
    logging.info('Starting the test 2 of load_data function ')

    df = None
    try:
        df = load_data('file_path.txt')
    except FileNotFoundError as err:
        logging.info('SUCCESS: FileNotFoundError is \
            expected in this test.')

    assert not df
    
    logging.info('SUCCESS: Test ended successfully')
    

@pytest.mark.parametrize("inference", [model.inference])
def test_inference(inference):
    """
    This function test the inference feature in the model.

    The test verify that the inference feature call the predict 
    function in the model parameter.

    """

    import numpy as np

    X = np.random.randint(-1,1,size=(5,3))

    class simple_model:

        def predict(self, X):

            return np.ones((X.shape[0],1))

    y_pred = inference(simple_model(), X)
    
    try:
        assert (X.shape[0],1) == y_pred.shape
    except AssertionError as err:
        logging.info('ERROR: Size mismatch between y_pred \
            and X.')
        raise err

    try:
        assert np.all(y_pred == 1)
    except AssertionError as err:
        logging.info('ERROR: The predict function in the model\
            parameter was not used.')
        raise err


@pytest.mark.parametrize('cmm', [model.compute_model_metrics])
def test_CMM(cmm):
    """
    This test is dedicated to compute_model_metrics function.    
    """       

    import numpy as np
    
    label = np.random.randint(0,1, size=(20,1))
    y_pred = np.random.randint(0,1, size=(20,1))
    results = cmm(label, y_pred)

    try:
        assert len(results) == 3
    except AssertionError as err:
        logging.info('ERROR: Some results are missing.')
        raise err

    try:
        results = np.array(results)
        assert np.all( results <= 1) and np.all( 0 <= results)
    except AssertionError as err:
        logging.info('ERROR: All the results should be \
        in [0, 1]')
        raise err