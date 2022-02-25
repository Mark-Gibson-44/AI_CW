import yaml
import pytest

f_name = 'config.yaml'

with open('config.yaml') as file:
        yaml_data= yaml.safe_load(file)

def test_yaml_content():

    
    assert len(yaml_data) != 0

def test_data_split():
    
    data_split_param = yaml_data['DATA_SPLIT']

    if type(data_split_param) == float:
        print("HERE")
        assert data_split_param > 0 and data_split_param < 0.5
    else:
        assert True

def test_smote_settings():
    lr = yaml_data['SMOTE']

    # Will fail before this line if Smote not within YAML
    assert True and type(lr) == bool


