from sklearn import metrics
import numpy as np
import pickle

class FeatureSelection:
	def __init__(self, features, traintestData, algoData, saveBestModel = False, param = 'testAccuracy', cutOff = 0.5, sampling = 10):
		self.features = features[0]
		self.target = features[1]
		self.cutOff = cutOff
		self.train_x = traintestData[0]
		self.train_y = traintestData[1]
		self.test_x = traintestData[2]
		self.test_y = traintestData[3]
		self.algoPath = algoData[0]
		self.algorithm = algoData[1]
		self.arguments = algoData[2]
		self.param = param
		self.sampling = sampling
		self.saveBestModel = saveBestModel
		self.featureAdditionsToAccuracy = []
		self.bestAccuracyGraph = []
		bestModel, bestAccuracy = self.trainModel(self.forwardSelection([], self.features), self.param, sampling = 1)
		with open('resultsFeatureSelection.txt','a') as f:
			print('The best accuracy obtained was',bestAccuracy)
			print('The best accuracy obtained was',bestAccuracy, file = f)
		if(saveBestModel):
			self.saveModel(bestModel, saveBestModel)

	def forwardSelection(self, selectedFeatures, featureList, currentAccuracy = 0):
		bestAccuracy = currentAccuracy
		bestFeature = None
		for feature in featureList:
			selectedFeaturesFinal = selectedFeatures.copy()
			selectedFeaturesFinal.append(feature)
			model, accuracy = self.trainModel(selectedFeaturesFinal, self.param, self.sampling)
			if(accuracy > bestAccuracy):
				bestAccuracy = accuracy
				bestFeature = feature

		if(bestFeature is not None):
			finalSelectedFeatures = selectedFeatures.copy()
			finalSelectedFeatures.append(bestFeature)
			featureList.remove(bestFeature)
			self.bestAccuracyGraph.append(bestAccuracy)
			self.featureAdditionsToAccuracy.append([bestFeature, bestAccuracy - currentAccuracy])
			return self.forwardSelection(finalSelectedFeatures, featureList, bestAccuracy)
		else:
			print('Results', featureList, selectedFeatures)
			return selectedFeatures

	def trainModel(self, selectedFeaturesFinal, param, sampling = 10):
		trainAccuracy = []
		testAccuracy = []
		truePositiveRate = []
		trueNegativeRate = []
		for samp in range(sampling):
			exec('from ' + self.algoPath + ' import ' + self.algorithm)
			# print(selectedFeaturesFinal)
			train_x = self.train_x[selectedFeaturesFinal]
			test_x = self.test_x[selectedFeaturesFinal]
			model = eval(self.algorithm + '(' + self.arguments + ')')
			model.fit(train_x, self.train_y)
			predictedResults = model.predict_proba(test_x)
			predictions = np.array([(-1 if x[0] >= self.cutOff else 1) for x in predictedResults])
			trainAccuracy.append(model.score(train_x,self.train_y))
			testAccuracy.append(model.score(test_x,self.test_y))
			truePositiveRate.append(metrics.precision_score(y_pred = predictions, y_true = self.test_y))
			if(sum(model.classes_) == 0):
				trueNegativeRate.append(metrics.precision_score(y_pred = -1*predictions, y_true = -1*self.test_y))
			else:
				trueNegativeRate.append(metrics.precision_score(y_pred = abs(predictions -1), y_true = abs(self.test_y - 1)))

		trainAccuracy = np.mean(trainAccuracy)
		testAccuracy = np.mean(testAccuracy)
		truePositiveRate = np.mean(truePositiveRate)
		trueNegativeRate = np.mean(trueNegativeRate)
		with open('resultsFeatureSelection.txt','a') as f:
			print(selectedFeaturesFinal, 'Train Accuracy: ' + '%.3f' % trainAccuracy + '\n' + 'Test Accuracy: ' + '%.3f' %  testAccuracy + '\n' + 'True Positive Rate: ' + '%.3f' %  truePositiveRate + '\n' + 'True Negative Rate: ' + '%.3f' %  trueNegativeRate + '\n\n\n\n')
			print(selectedFeaturesFinal, 'Train Accuracy: ' + '%.3f' % trainAccuracy + '\n' + 'Test Accuracy: ' + '%.3f' %  testAccuracy + '\n' + 'True Positive Rate: ' + '%.3f' %  truePositiveRate + '\n' + 'True Negative Rate: ' + '%.3f' %  trueNegativeRate + '\n\n\n\n', file = f)
		toReturn = eval(param)		
		return model,toReturn

	def saveModel(self,model, filename):
		with open(filename, 'wb') as f:
			pickle.dump(model, f)