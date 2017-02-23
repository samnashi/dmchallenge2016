'''
Use trained model to predict image in crosswalk file
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sc1_infer.py <in:dcm dir> <out:temp output dir> <in:crosswalk file> <in:model architecture json file> <in:model weights h5 file> <out:prediction result tsv>

'''
import sys
import os
import csv
import numpy as np
from keras.models import model_from_json
from preprocess import preprocess_image, EXPECTED_DIM, MAX_VALUE, FILTER_THRESHOLD

PREDICTIONS_PATH = 'predictions.tsv'

dcm_dir = sys.argv[1]
scratch_dir = sys.argv[2]
crosswalk_file = sys.argv[3]
arch_file = sys.argv[4]
weights_file = sys.argv[5]
predictions_file = sys.argv[6] if len(sys.argv) > 6 else PREDICTIONS_PATH

# load model
print('Loading model')
with open(arch_file) as f:
    arch_json = f.read()
    model = model_from_json(arch_json)
model.load_weights(weights_file)

predictions = {}
prediction_index = []
batch_key_list = []

mini_batch_size = 100 #reference
final_mini_batch_size=10 #placeholder, but it really is calculated at the end of the penultimate batch
dataTest = np.zeros((mini_batch_size, 3, 299, 299), dtype=np.uint16)

mini_batch=mini_batch_size #this is the batch counter


print('Predicting by batches of {}'.format(mini_batch_size))

with open(crosswalk_file, 'rb') as tsvin:
    crosswalk = csv.reader(tsvin, delimiter='\t')
    headers = next(crosswalk, None)
    row_count = sum(1 for row in crosswalk)
    print('row count: {}'.format(row_count))
tsvin.close()

with open(crosswalk_file, 'rb') as tsvin:
    crosswalk = csv.reader(tsvin, delimiter='\t')
    headers = next(crosswalk, None)
    count = 1
    print('row count: {}'.format(row_count))
    isFinal = False
    isPenultimate=False
    for row in crosswalk:
        # no exam id col for testing
        dcm_subject_id = row[0]
        dcm_laterality = row[3]
        dcm_filename = row[4]
        # dcm_laterality = row[4]
        # dcm_filename = row[5]
        data_points_left = row_count

        data = preprocess_image(os.path.join(dcm_dir, dcm_filename), dcm_laterality)
        if mini_batch != 0:
            #print(dcm_filename)
            if isPenultimate == True: #final batch of dataTest will have the smaller final batch size
                dataTest = np.zeros((final_mini_batch_size, 3, 299, 299), dtype=np.uint16)

            dataTest[mini_batch_size-mini_batch,0,:,:]=data[:,0,:,:]
            dataTest[mini_batch_size-mini_batch,1,:,:]=data[:,1,:,:]
            dataTest[mini_batch_size-mini_batch,2,:,:]=data[:,2,:,:]
            key = '{}_{}'.format(dcm_subject_id, dcm_laterality)
            #print('key: ',key)
            #if key in predictions: print('Duplicate detected: {}'.format(key))
            if key not in predictions: #if key isn't in the master DICT, add it
                predictions[key] = {
                    'id': dcm_subject_id,
                    'lat': dcm_laterality,
                    'p': []
                }
                prediction_index.append(key) #prediction_index is for writing later, to ensure no duplication.
            #print('pred index: ', prediction_index) 
            batch_key_list.append(key) #OF THE BATCH, same size as prediction. This gets cleared!
            #print('batch key list: '.format(batch_key_list))
            mini_batch-=1
            print ('count: {} mini_batch: {} key: {}'.format(count, mini_batch,key))
            data_points_left = (row_count - (count + 1))
            print('In if minibatch!=0 if, data points left: {}'.format(data_points_left))
        if mini_batch == 0  and isFinal == False: #if you've hit the end of the batch.
            print ('dataTest shape: {} ndim: {} size: {}'.format(dataTest.shape, dataTest.ndim, dataTest.size))
            data_points_left=(row_count-(count+1))
            print('In if minibatch==0 if, data points left: {}'.format(data_points_left))
            #gotta do this twice. set up the last batch and load the last batch's result..

            #penultimate batch
            if ((data_points_left < mini_batch_size) and isFinal == False and data_points_left > 0 and isPenultimate == False): #special case for the 2nd to last one..  & row_count-(count+1) >= 0
                print('Penultimate IF entered with count+1:{} '
                    'row count:{} calculated '
                    'final batch size:{}'.format(count+1,row_count,row_count-(count+1)))
                final_mini_batch_size = data_points_left
                #mini_batch_size = final_mini_batch_size #yes i know it's hack but this is what is passed to model.predict()
                mini_batch = final_mini_batch_size # reset batch counter so it's the final batch size GETS PASSED UP TOP
                isPenultimate=True #flag signifying this is the final batch to predict
                print("penultimate minibatch size: {} isPenultimate:{}".format(len(prediction),isPenultimate))
                prediction = model.predict(dataTest, batch_size=mini_batch_size)
                print("penultimate prediction length: {}".format(len(prediction)))
                mini_batch_size = final_mini_batch_size #SET THIS TO THE FINAL SIZE TO SET UP FOR THE FINAL ITERATION. GETS PASSED UP TOP

            #ultimate batch is when data point left is 0. This gets entered at the END of the penultimate batch
            if ((data_points_left < mini_batch_size) and isFinal==False and isPenultimate==True): #special case for the last one..  & row_count-(count+1) >= 0
                print('Final IF entered with count+1:{} '
                    'row count:{} calculated '
                    'final batch size:{}'.format(count+1,row_count,row_count-(count+1)))
                #final_mini_batch_size = data_points_left
                #mini_batch_size = final_mini_batch_size #yes i know it's hack but this is what is passed to model.predict()
                #mini_batch = final_mini_batch_size # reset batch counter so it's the final batch size
                isFinal=True #flag signifying this is the final batch to predict
                prediction = model.predict(dataTest, batch_size=final_mini_batch_size) #don't update the batch size since it'll be 0, using the previous metric..
                print("final prediction length: {} finalbatchsize: {}".format(len(prediction),final_mini_batch_size))

            if ((row_count-(count+1)) >= mini_batch_size and isFinal==False and isPenultimate==False and data_points_left > 0):
                print('normal IF entered. count:{}'.format(count))
                mini_batch = mini_batch_size
                prediction = model.predict(dataTest, batch_size=mini_batch_size)
                print("Normal prediction length: {}".format(len(prediction)))

            if isFinal == True and isPenultimate == True:
                print("FINAL BATCH CHECK. Length: {}, Prediction  {}".format(len(prediction),prediction))
            #print ('Prediction shape: {} ndim: {} size: {}'.format(prediction.shape, prediction.ndim, prediction.size))
            #loads into the dict
            #print('batch key list: '.format(batch_key_list))
            for i in range(0,len(prediction)-1): #writes batch keys matched with batch prediction results #range(0,len(prediction)-1)
                #if there's something under the batch key that's already in the dict, it'll be appended correctly
                #print ('before isfinite(), predictions{} is {}'.format(batch_key_list[i], prediction[i][0]))
                #if np.isfinite(prediction[i][0])==True:
                print('batch_key_list[i]:{}'.format(batch_key_list[i]))
                predictions[batch_key_list[i]]['p'].append(prediction[i][0])
                #print ('after isfinite(), predictions{} is {}'.format(batch_key_list[i], prediction[i][0]))
            batch_key_list=[] #empty out batch keys after being written
            for key in prediction_index: #just prints for diagnostics. 
                pred = predictions[key]
                #print ('For key {}'.format(key))
                #print ('predictions{} is {}'.format(key,pred))
                #print ('predictions[key][p] is {}'.format(pred['p']))
                #print('--------------------------------------')
        count += 1


    #write predictions
print('Count: {}'.format(count))
print('Writing to result {}'.format(predictions_file))
with open(predictions_file, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t')
    header = ('subjectId', 'laterality', 'confidence')
    spamwriter.writerow(['subjectId', 'laterality', 'confidence'])
    # iterate index to keep order
    for key in prediction_index: #key in prediction_index
        pred = predictions[key]
        if len(pred['p']) > 0: #skip those with no predictions.. disable temporarily so i get a log
            print ('predictions[key][p] is {}'.format(pred['p']))
            #confidence = sum(pred['p']) / float(len(pred['p'])) #for average!
            #confidence=pred['p'][0] #the first prediction
            confidence = max(pred['p'])
        print ('confidence is {}, len(pred[p]) is {}'.format(confidence,len(pred['p'])))
        if len(pred['p']) == 0: #if there are no predictions, put in a 0.05 to guard against batch size spillover?
            confidence = 0.15
        row = (pred['id'], pred['lat'], confidence)
        spamwriter.writerow(row)
print('Done.')
