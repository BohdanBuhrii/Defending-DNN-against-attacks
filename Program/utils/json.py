import json
import numpy as np

def save_to_json(parameters, file_name):
  serializable_dict = {}
  for key in parameters.keys():
    serializable_dict[key] = parameters[key].tolist()

  with open(file_name + ".json", "w") as write_file:
    json.dump(serializable_dict, write_file)


def read_from_json(file_name):
  deserialized_dict = {}
  with open(file_name + ".json", "r") as read_file:
    deserialized_dict = json.load(read_file)

  parameters = {}
  for key in deserialized_dict.keys():
    parameters[key] = np.array(deserialized_dict[key])

  return parameters
