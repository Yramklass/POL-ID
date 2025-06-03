import timm 
available_coatnets = timm.list_models('coatnet*')
print(available_coatnets)