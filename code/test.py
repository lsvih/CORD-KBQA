from train import train
from configure import base_path
import utils
import os


if __name__ == '__main__':
	bm = 'sq'
	all_path = os.listdir('./runs/'+bm+'/l2_rtype')
	outputs = []
	corrects = []
	for path in all_path:
		if not os.path.isdir(os.path.join('./runs/'+bm+'/l2_rtype', path)):
			continue
		print(path)
		pred_path = './runs/'+bm+'/l2_rtype/'+path+'/predictions.txt'
		correct_path = './runs/'+bm+'/l2_rtype/'+path+'/corrected.txt'
		# qid ques rid rel correct
		preds = [line.split('\t')[0] + '\t' + line.split('\t')[2] for line in utils.loadLists(pred_path)]
		corrects += utils.loadLists(correct_path)[:-1]
		outputs = outputs + preds
	utils.writeList('./runs/'+bm+'/l2_rtype/all_preds.txt', outputs)
	utils.writeList('./runs/'+bm+'/l2_rtype/all_corrected.txt', corrects)