from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score


def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])


 
    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions = baseline.test(data.devset)

    y_pred = []
    for e in predictions:
        y_pred.append(str(int(e)))

    gold_labels = [sent['gold_label'] for sent in data.devset]

    report_score(gold_labels, y_pred)


if __name__ == '__main__':
    #execute_demo('english')
    execute_demo('spanish')
    execute_demo('english')


