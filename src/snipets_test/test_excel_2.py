import pandas as pd
import xlsxwriter
from classes.ResultsHandlerExcel import ResultsHandlerExcel


experiments =  ['delta_experiment_next_period_1', 
                'delta_experiment_next_period_2', 
                'delta_experiment_next_period_3', 
                'delta_experiment_next_period_4', 
                'delta_experiment_next_period_5']

sets = ['validation', 'test']

initial_validation_row_index = 1
initial_test_row_index = 1

# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('demo_scheda_unica.xlsx')

worksheet = workbook.add_worksheet('Return')

for set in sets:

    for index, experiment in enumerate(experiments):
        
        results_handler = ResultsHandlerExcel(experiment_name=experiment, dataset='sp500')

        if set == 'validation':
            df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/' + experiment + '/results/ensemble/' + set + '/validation_all_thr_results.csv')
        
            header = [{'header': di} for di in df.columns.values]

            worksheet.write('A' + str(initial_validation_row_index), 'Return Validation - Run ' + str(index+1))
            worksheet.add_table('A' + str(initial_validation_row_index+1) + ':H' + str(initial_validation_row_index+7), {  'data': df.values.tolist(),
                                            'columns': header,
                                            'header_row': True,
                                        })

            initial_validation_row_index = initial_validation_row_index+10

        if set == 'test':
            df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/' + experiment + '/results/ensemble/' + set + '/test_all_thr_results.csv')


            header = [{'header': di} for di in df.columns.values]
            worksheet.write('K' + str(initial_test_row_index), 'Return Test - Run ' + str(index+1))
            worksheet.add_table('K' + str(initial_test_row_index+1) + ':R' + str(initial_test_row_index+7), {  'data': df.values.tolist(),
                                            'columns': header,
                                            'header_row': True,
                                        })

            initial_test_row_index = initial_test_row_index+10


initial_validation_row_index = 1
initial_test_row_index = 1

for set in sets:
    for index, experiment in enumerate(experiments):

        results_handler = ResultsHandlerExcel(experiment_name=experiment, dataset='sp500')

        if set == 'validation':
            df_accuracy, df_coverage = results_handler.generate_ensemble_and_plots(set_type=set)
            #print(df_accuracy)
            
            header = [{'header': di} for di in df_accuracy.columns.values]

            worksheet.write('U' + str(initial_validation_row_index), 'Accuracy Validation - Run ' + str(index+1))
            worksheet.add_table('U' + str(initial_validation_row_index+1) + ':AA' + str(initial_validation_row_index+7), {  'data': df_accuracy.values.tolist(),
                                            'columns': header,
                                            'header_row': True,
                                        })
            
            initial_validation_row_index = initial_validation_row_index+10
        
        if set == 'test':
            df_accuracy, df_coverage = results_handler.generate_ensemble_and_plots(set_type=set)

            header = [{'header': di} for di in df_accuracy.columns.values]
            worksheet.write('AD' + str(initial_test_row_index), 'Accuracy Test - Run ' + str(index+1))
            worksheet.add_table('AD' + str(initial_test_row_index+1) + ':AJ' + str(initial_test_row_index+7), {  'data': df_accuracy.values.tolist(),
                                            'columns': header,
                                            'header_row': True,
                                        })

            initial_test_row_index = initial_test_row_index+10


initial_validation_row_index = 1
initial_test_row_index = 1

for set in sets:

    for index, experiment in enumerate(experiments):
        results_handler = ResultsHandlerExcel(experiment_name=experiment, dataset='sp500')

        if set == 'validation':
            df_accuracy, df_coverage = results_handler.generate_ensemble_and_plots(set_type=set)
            header = [{'header': di} for di in df_coverage.columns.values]
            worksheet.write('AM' + str(initial_validation_row_index), 'Coverage Validation - Run ' + str(index+1))
            worksheet.add_table('AM' + str(initial_validation_row_index+1) + ':AS' + str(initial_validation_row_index+7), {  'data': df_coverage.values.tolist(),
                                            'columns': header,
                                            'header_row': True,
                                        })
            initial_validation_row_index = initial_validation_row_index+10            
        if set == 'test':
            df_accuracy, df_coverage = results_handler.generate_ensemble_and_plots(set_type=set)

            header = [{'header': di} for di in df_coverage.columns.values]
            worksheet.write('AV' + str(initial_test_row_index), 'Coverage Test - Run ' + str(index+1))
            worksheet.add_table('AV' + str(initial_test_row_index+1) + ':BB' + str(initial_test_row_index+7), {  'data': df_coverage.values.tolist(),
                                            'columns': header,
                                            'header_row': True,
                                        })

            initial_test_row_index = initial_test_row_index+10

workbook.close()




