import argparse, torch, os, time, json
from gradescope_utils.autograder_utils.decorators import weight
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import chess.lib.heuristics
import extracredit_embedding
import extracredit

def compute_winratio(subset):
    '''
    winratio = compute_winratio(filename) -
        Run a competition between your model and chess.lib.heuristics.evaluate, 
        to see which one better predicts the value assigned to the board in the test set.
    Input: 
       filename: the name of a file in which each line is a test case, as a JSON string
    Output: 
       winratio - the fraction of test tokens on which your model outperformed the baseline.
    '''
    filename = 'extracredit_%s.txt'%(subset)
    if not os.path.isfile(filename):
        return -1
    dataset = extracredit_embedding.ChessDataset(filename)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000)
    model = torch.load('model_ckpt.pkl')
        
    score = 0
    for x,y in dataloader:
        hypVal = model(x)
        for t in range(1000):
            side, board, flags = extracredit_embedding.unembed_board(x.numpy()[t])
            advVal = chess.lib.heuristics.evaluate(board)
        
            if abs(hypVal[t].item()-y[t].item()) < abs(advVal-y[t].item()):
                score += 1
            elif abs(hypVal[t].item()-y[t].item()) == abs(advVal-y[t].item()):
                score += 0.5
    return score / 1000

def create_report(winratio, elapsedtime, subsets):
    grade_ratio = (winratio['validation'] if winratio['evaluation'] == -1 else winratio['evaluation'])
    score =(2 * (grade_ratio >= 0.5)) + 100 * min(0.08, max(grade_ratio - 0.5, 0)) 
    report = {
        'tests' : [
            {
                'name':'validoreval_winratio_gt_0.5',
                'score': score,
                'max_score': 10
            }
        ],
        'visibility' : 'visible',
        'execution_time' : elapsedtime,
        'score' : score,
        'leaderboard' : [
            { 'name': 'winratio_evaluation', 'value' : winratio['evaluation'] },
            { 'name': 'winratio_validation', 'value' : winratio['validation'] },
            { 'name': 'Time', 'value': elapsedtime },
        ]
    }
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grade the extra credit part.')
    parser.add_argument('--profiler',action='store_true',
                        help='''Run profiler, to see how long each test is taking.''')
    args = parser.parse_args()

    start = time.time()
    subsets = ['validation', 'evaluation']
    winratio = {}
    for subset in subsets:
        winratio[subset] = compute_winratio(subset)
    elapsedtime = time.time() - start

    report = create_report(winratio, elapsedtime, subsets)

    print(json.dumps(report, indent=2))
