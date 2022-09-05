import pandas as pd


def save_to_csv(results, save_name=None):
    column_names = ['Clean', 'FGSM ', 'PGD-10', 'PGD-20',
                    'A(FGSM)', 'B(FGSM)', 'C(FGSM)', 'D(FGSM)',
                    'A(PGD-10)', 'B(PGD-10)', 'C(PGD-10)', 'D(PGD-10)'
                    'A(PGD-20)', 'B(PGD-20)', 'C(PGD-20)', 'D(PGD-20)']
    df = pd.DataFrame(results, columns=column_names)
    df.to_csv("{}.csv".format(save_name))
