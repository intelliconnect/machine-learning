import h2o
import os
import tabulate
import operator
from h2o.estimators.gbm import H2OGradientBoostingEstimator


h2o.init()

#Loading Data
productionprocess = h2o.import_file(path=os.path.realpath("/home/iconnect4/bespoke_manufacturing/data/production-process-data.csv"),destination_frame = "bespokemanufacturing",header=1,col_types=["string","string","string","string","string","string","string"])
productionprocess.describe()


data_cols = ["productshortname","prodordertype","prodordercategory","orderitempriority","ordersource","address_dl_country","prod_allocated_process"]
for col in data_cols :
    productionprocess[col] = productionprocess[col].asfactor()


#Split into train and test frames
train,  test =  productionprocess.split_frame(ratios=[0.7])
print(train.nrows)
print(test.nrows)


#Predictor and Response columns
predictor_columns = productionprocess.names[:]
predictor_columns.remove("prod_allocated_process")
response_column = "prod_allocated_process"


#Building the GBM model
gbm_model = H2OGradientBoostingEstimator(ntrees = 50,max_depth = 6,learn_rate   = 0.1,distribution= 'auto' )
gbm_model.train(x = predictor_columns,y = response_column,training_frame = train, validation_frame = test)

# print(gbm_model)
# g=gbm_model.model_performance(test)
# print(g)

# Download MOJO
modelfile = gbm_model.download_mojo(path="/home/iconnect4/bespoke_manufacturing/productionprocessprediction", get_genmodel_jar=True)


# Validate Model
# Vest
testdata1 = {"productshortname" : ["V"],
     "prodordertype":["FU"],
     "prodordercategory": ["F"],
     "orderitempriority" : ["Normal"],
     "ordersource" : ["SS"],
     "address_dl_country" : ["United States of America (the)"]
         }

# Jacket
testdata2 = {"productshortname" : ["J"],
     "prodordertype":["FU"],
     "prodordercategory": ["N"],
     "orderitempriority" : ["Normal"],
     "ordersource" : ["SS"],
     "address_dl_country" : ["Netherlands (the)"]
         }

# Skirt
testdata3 = {"productshortname" : ["SK"],
     "prodordertype":["FU"],
     "prodordercategory": ["N"],
     "orderitempriority" : ["Normal"],
     "ordersource" : ["SS"],
     "address_dl_country" : ["United States of America (the)"]
         }

# Shirt
testdata4 = {"productshortname" : ["SH"],
     "prodordertype":["FU"],
     "prodordercategory": ["R"],
     "orderitempriority" : ["Normal"],
     "ordersource" : ["SS"],
     "address_dl_country" : ["United States of America (the)"]
         }


#Creating H2O frame for example
testdata1example = h2o.H2OFrame(testdata1)
testdata2example = h2o.H2OFrame(testdata2)
testdata3example = h2o.H2OFrame(testdata3)
testdata4example = h2o.H2OFrame(testdata4)

#Predicting GBM outcome for the examples
gbm_pred= gbm_model.predict(testdata1example)
gbm_pred2= gbm_model.predict(testdata2example)
gbm_pred3= gbm_model.predict(testdata3example)
gbm_pred4= gbm_model.predict(testdata4example)

#Displays all the entries in the response column
f= productionprocess['prod_allocated_process'].levels()
print(f)

#To display the gbm predictions
print("GBM PREDICTIONS")
output  = [
           [testdata1["productshortname"][0],100*gbm_pred],
           [testdata2["productshortname"][0],100*gbm_pred2],
           [testdata3["productshortname"][0], 100 * gbm_pred3],
           [testdata4["productshortname"][0], 100 * gbm_pred4]
         ]

h2o.display.H2ODisplay(output)

