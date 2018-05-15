from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score


def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])


    #samples = [1500,3000,6000,9000,12000,13750] #Spanish
    samples = [5000,10000,15000,20000,25000,27299] #English
    baseline = Baseline(language)
    for i in samples:
        print(len(data.trainset[:i]))
        baseline.train(data.trainset[:i])

        gold_labels = [sent['gold_label'] for sent in data.devset]

        predictions = baseline.test(data.devset,gold_labels)

        y_pred = []
        for e in predictions:
            y_pred.append(str(int(e)))

    #gold_labels = [sent['gold_label'] for sent in data.devset]
        print(i)
        report_score(gold_labels, y_pred)
    
    #print("Test Data:")
    #gold_labels = [sent['gold_label'] for sent in data.testset]
    #baseline.train(data.testset)
    #predictions = baseline.test(data.testset,gold_labels)

   # y_pred = []
   # for e in predictions:
      # y_pred.append(str(int(e)))
    #report_score(gold_labels, y_pred)
    


if __name__ == '__main__':
    #execute_demo('spanish')
    execute_demo('english')


