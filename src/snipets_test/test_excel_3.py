import pandas as pd
import numpy as np
import xlsxwriter
from classes.ResultsHandlerExcel import ResultsHandlerExcel

from string import ascii_uppercase
import itertools


#  
def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_uppercase, repeat=size):
            yield "".join(s)

def gen_list(max='ZZ'):
    list = []
    list.append(" ")

    for s in iter_all_strings():
        list.append(s)
        if s == max:
            break
    return list

def gen_header_table(workbook, worksheet, headers, row_index, column_index, run_index):
    cell_format_3 = workbook.add_format({'bold': True, 'align': 'left', 'border': 1, 'border_color': '#000000'})
    cell_format = workbook.add_format({'bold': True, 'align': 'center', 'border': 1, 'border_color': '#000000'})
    cell_format_2 = workbook.add_format({'bold': True, 'align': 'center', 'border': 1, 'border_color': '#000000', 'bg_color': '#cccccc'})
    # Se parto dalla colonna 1 -> 1 + [(thr + 5 walk + walk_all) * 3] - 1 perché thr ha una sola subcolonna (dovrebbe essere -2 ma ho il +1 della colonna iniziale)
    end_column = column_index + (len(headers) * 3) - 5

    worksheet.merge_range(excel_column_list[column_index] + str(row_index) + ':' + excel_column_list[end_column] + str(row_index), 'Run ' + str(run_index+1), cell_format_3)

    row_index = row_index + 1

    for header in headers:
        #if column_index == 1 or column_index == 25:
        if header == 'thr':
            worksheet.write(excel_column_list[column_index] + str(row_index), header, cell_format)
            worksheet.write(excel_column_list[column_index] + str(row_index + 1), " ", cell_format_2)
            column_index = column_index + 1
        else:
            # COLONNA walk_all
            if header == 'walk_all':
                worksheet.write(excel_column_list[column_index] + str(row_index), header, cell_format)
            else:
                worksheet.merge_range(excel_column_list[column_index] + str(row_index) + ':' + excel_column_list[column_index+2] + str(row_index), header, cell_format)

            worksheet.write(excel_column_list[column_index] + str(row_index + 1), "Return", cell_format)
            if header != 'walk_all':
                worksheet.write(excel_column_list[column_index + 1]  + str(row_index + 1), "Accuracy", cell_format)
                worksheet.write(excel_column_list[column_index + 2]  + str(row_index + 1), "Coverage", cell_format)

            column_index = column_index + 3

def fill_return_table(workbook, worksheet, df, df_accuracy, df_coverage,  row_index, column_index):
    cell_format = workbook.add_format({'bold': True, 'align': 'right', 'border': 1, 'border_color': '#000000'})
    cell_format_2 = workbook.add_format({'bold': False, 'align': 'center', 'border': 1, 'border_color': '#000000'})

    headers = df.columns.values

    column_cicle = column_index
    # COLONNE
    for index_header, header in enumerate(headers):
        
        # RIGHE 
        for index, row in df.iterrows():
            if(header == 'thr'):  
                worksheet.write(excel_column_list[column_cicle] + str(row_index + index + 3), row[header], cell_format)
            else:
                worksheet.write(excel_column_list[column_cicle] + str(row_index + index + 3), row[header], cell_format_2)

        if header != 'walk_all':
            for index, row in df_accuracy.iterrows():
                if header != 'thr':  
                    worksheet.write(excel_column_list[column_cicle+1] + str(row_index + index + 3), row[header], cell_format_2)

            for index, row in df_coverage.iterrows():
                if header != 'thr':  
                    worksheet.write(excel_column_list[column_cicle+2] + str(row_index + index + 3), row[header], cell_format_2)

        if column_cicle == column_index:      
            column_cicle = column_cicle + 1
        else: 
            column_cicle = column_cicle + 3
                    
##################################################################

excel_column_list = gen_list('LL')

#experiments =  ['delta_experiment_next_period_1', 
#                'delta_experiment_next_period_2', 
#                'delta_experiment_next_period_3', 
#                'delta_experiment_next_period_4', 
#                'delta_experiment_next_period_5']

experiments = ['3_months_delta_experiment_1', 
                '3_months_delta_experiment_2',
                '3_months_delta_experiment_3',
                '3_months_delta_experiment_4',
                '3_months_delta_experiment_5']


sets = ['validation', 'test']


initial_validation_row_index = 1
initial_test_row_index = 1

# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('demo_scheda_unica_new.xlsx')

worksheet = workbook.add_worksheet('Full Report')

for set in sets:

    row_index = 1

    for index, experiment in enumerate(experiments):
        
        results_handler = ResultsHandlerExcel(experiment_name=experiment, dataset='sp500')
        
        
        df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/' + experiment + '/results/ensemble/' + set + '/' + set + '_all_thr_results.csv')
        df_accuracy, df_coverage = results_handler.generate_ensemble_and_plots(set_type=set)

        if set == 'validation':
            headers = df.columns.values

            gen_header_table(workbook=workbook, 
                            worksheet=worksheet, 
                            headers=headers, 
                            row_index=row_index, 
                            column_index=1, 
                            run_index=index)

            fill_return_table(  workbook=workbook, 
                                worksheet=worksheet, 
                                df=df, 
                                df_accuracy=df_accuracy, 
                                df_coverage=df_coverage,  
                                row_index=row_index, 
                                column_index=1)

        if set == 'test':
            headers = df.columns.values

            gen_header_table(workbook=workbook, 
                            worksheet=worksheet, 
                            headers=headers, 
                            row_index=row_index, 
                            column_index=108, #23
                            run_index=index)

            fill_return_table(  workbook=workbook, 
                                worksheet=worksheet, 
                                df=df, 
                                df_accuracy=df_accuracy, 
                                df_coverage=df_coverage,  
                                row_index=row_index, 
                                column_index=108) #23

        row_index = row_index + 11

workbook.close()     
            #worksheet.add_table('A' + str(initial_validation_row_index+1) + ':H' + str(initial_validation_row_index+7), {  'data': df.values.tolist(),
                                    #        'columns': header,
                                  #          'header_row': True,
                                    #    })

           # initial_validation_row_index = initial_validation_row_index+10

       # if set == 'test':
         #   df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/' + experiment + '/results/ensemble/' + set + '/test_all_thr_results.csv')


           # header = [{'header': di} for di in df.columns.values]
           # worksheet.write('K' + str(initial_test_row_index), 'Return Test - Run ' + str(index+1))
           # worksheet.add_table('K' + str(initial_test_row_index+1) + ':R' + str(initial_test_row_index+7), {  'data': df.values.tolist(),
            #                                'columns': header,
             #                               'header_row': True,
            #                            })

           # initial_test_row_index = initial_test_row_index+10

'''
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




'''

