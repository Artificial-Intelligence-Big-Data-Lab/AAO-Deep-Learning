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

def gen_header_table(workbook, worksheet, row_index, column_index, run_index, walk_index):
    cell_format_3 = workbook.add_format({'bold': True, 'align': 'center', 'border': 1, 'border_color': '#000000'})
    cell_format = workbook.add_format({'bold': True, 'align': 'center', 'border': 1, 'border_color': '#000000'})
    cell_format_2 = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'center', 'border': 1, 'border_color': '#000000', 'bg_color': '#cccccc'})

    # NUMERO RUN
    worksheet.merge_range(excel_column_list[column_index + 1] + str(row_index) + ':' + excel_column_list[column_index + 18] + str(row_index), 'Run ' + str(run_index), cell_format_3)

    # NUMERO WALK
    worksheet.merge_range(excel_column_list[column_index] + str(row_index) + ':' + excel_column_list[column_index] + str(row_index + 1), 'Walk ' + str(walk_index), cell_format_2)

    row_index = row_index + 1

    # HEADER PER VALIDATION - TEST
    worksheet.merge_range(excel_column_list[column_index + 1] + str(row_index) + ':' + excel_column_list[column_index + 6] + str(row_index), 'Validation ', cell_format_3)
    worksheet.merge_range(excel_column_list[column_index + 7] + str(row_index) + ':' + excel_column_list[column_index + 9] + str(row_index), 'Validation Buy Hold', cell_format_3)
    worksheet.merge_range(excel_column_list[column_index + 10] + str(row_index) + ':' + excel_column_list[column_index + 15] + str(row_index), 'Test ', cell_format_3)
    worksheet.merge_range(excel_column_list[column_index + 16] + str(row_index) + ':' + excel_column_list[column_index + 18] + str(row_index), 'Test Buy Hold ', cell_format_3)

    row_index = row_index + 1

    # HEADER CON THR E VARI PARAMETRI DA RIEMPIRE
    worksheet.write(excel_column_list[column_index] + str(row_index), "thr", cell_format)

    # VALIDATION 
    worksheet.write(excel_column_list[column_index + 1] + str(row_index), "return", cell_format)
    worksheet.write(excel_column_list[column_index + 2] + str(row_index), "accuracy", cell_format)
    worksheet.write(excel_column_list[column_index + 3] + str(row_index), "coverage", cell_format)
    worksheet.write(excel_column_list[column_index + 4] + str(row_index), "trade count", cell_format)
    worksheet.write(excel_column_list[column_index + 5] + str(row_index), "mdd", cell_format)
    worksheet.write(excel_column_list[column_index + 6] + str(row_index), "romad", cell_format)

    # VALIDATION BH
    worksheet.write(excel_column_list[column_index + 7] + str(row_index), "return", cell_format)
    worksheet.write(excel_column_list[column_index + 8] + str(row_index), "mdd", cell_format)
    worksheet.write(excel_column_list[column_index + 9] + str(row_index), "romad", cell_format)

    # TEST
    worksheet.write(excel_column_list[column_index + 10] + str(row_index), "return", cell_format)
    worksheet.write(excel_column_list[column_index + 11] + str(row_index), "accuracy", cell_format)
    worksheet.write(excel_column_list[column_index + 12] + str(row_index), "coverage", cell_format)
    worksheet.write(excel_column_list[column_index + 13] + str(row_index), "trade count", cell_format)
    worksheet.write(excel_column_list[column_index + 14] + str(row_index), "mdd", cell_format)
    worksheet.write(excel_column_list[column_index + 15] + str(row_index), "romad", cell_format)

    # TEST BH
    worksheet.write(excel_column_list[column_index + 16] + str(row_index), "return", cell_format)
    worksheet.write(excel_column_list[column_index + 17] + str(row_index), "mdd", cell_format)
    worksheet.write(excel_column_list[column_index + 18] + str(row_index), "romad", cell_format)
    


def fill_return_table(workbook, worksheet, 
                validation_df, validation_df_accuracy, validation_df_coverage, validation_df_trade_count, validation_df_mdd, validation_df_romad,
                test_df,  test_df_accuracy, test_df_coverage, test_df_trade_count, test_df_mdd, test_df_romad,
                validation_bh_df_return, validation_bh_df_romad, validation_bh_df_mdd,
                test_bh_df_return, test_bh_df_romad, test_bh_df_mdd,
                row_index, column_index):
    cell_format = workbook.add_format({'bold': True, 'align': 'right', 'border': 1, 'border_color': '#000000'})
    cell_format_2 = workbook.add_format({'bold': False, 'align': 'center', 'border': 1, 'border_color': '#000000'})

    thrs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for index, thr in enumerate(thrs): 
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), thr, cell_format)

    column_index += 1
    
    # VALIDATION RETURN
    for index, row in enumerate(validation_df):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # VALIDATION ACCURACY
    column_index += 1
    for index, row in enumerate(validation_df_accuracy):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)
    
    # VALIDATION COVERAGE
    column_index += 1
    for index, row in enumerate(validation_df_coverage):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # VALIDATION TRADE COUNT
    column_index += 1
    for index, row in enumerate(validation_df_trade_count):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)
    
    # VALIDATION MDD
    column_index += 1
    for index, row in enumerate(validation_df_mdd):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # VALIDATION MDD
    column_index += 1
    for index, row in enumerate(validation_df_romad):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    column_index += 1
    # USO IL CICLO DELLE SOGLIE PER PRINTARE LOS TESSO VALORE SU PIU' RIGHE DI BH
    for index, thr in enumerate(thrs): 
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), validation_bh_df_return, cell_format_2)
        worksheet.write(excel_column_list[column_index + 1] + str(row_index + index + 3), validation_bh_df_mdd, cell_format_2)
        worksheet.write(excel_column_list[column_index + 2] + str(row_index + index + 3), validation_bh_df_romad, cell_format_2)

     # TEST RETURN
    column_index += 3
    for index, row in enumerate(test_df):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # TEST ACCURACY
    column_index += 1
    for index, row in enumerate(test_df_accuracy):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)
    
    # TEST COVERAGE
    column_index += 1
    for index, row in enumerate(test_df_coverage):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # TEST TRADE COUNT
    column_index += 1
    for index, row in enumerate(test_df_trade_count):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # TEST TRADE COUNT
    column_index += 1
    for index, row in enumerate(test_df_mdd):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)
    
    # TEST TRADE COUNT
    column_index += 1
    for index, row in enumerate(test_df_romad):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    column_index += 1
    # USO IL CICLO DELLE SOGLIE PER PRINTARE LOS TESSO VALORE SU PIU' RIGHE DI BH
    for index, thr in enumerate(thrs): 
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), test_bh_df_return, cell_format_2)
        worksheet.write(excel_column_list[column_index + 1] + str(row_index + index + 3), test_bh_df_mdd, cell_format_2)
        worksheet.write(excel_column_list[column_index + 2] + str(row_index + index + 3), test_bh_df_romad, cell_format_2)


def fill_return_table_global(workbook, worksheet, 
                validation_df, validation_df_accuracy, validation_df_coverage, validation_df_trade_count, validation_df_mdd, validation_df_romad,
                test_df,  test_df_accuracy, test_df_coverage, test_df_trade_count, test_df_mdd, test_df_romad,
                validation_bh_df_return, validation_bh_df_romad, validation_bh_df_mdd,
                test_bh_df_return, test_bh_df_romad, test_bh_df_mdd,
                row_index, column_index):
    cell_format = workbook.add_format({'bold': True, 'align': 'right', 'border': 1, 'border_color': '#000000'})
    cell_format_2 = workbook.add_format({'bold': False, 'align': 'center', 'border': 1, 'border_color': '#000000'})

    thrs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for index, thr in enumerate(thrs): 
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), thr, cell_format)

    column_index += 1
    
    # VALIDATION RETURN
    for index, row in enumerate(validation_df):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # VALIDATION ACCURACY
    column_index += 1
    for index, row in enumerate(validation_df_accuracy):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)
    
    # VALIDATION COVERAGE
    column_index += 1
    for index, row in enumerate(validation_df_coverage):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # VALIDATION TRADE COUNT
    column_index += 1
    for index, row in enumerate(validation_df_trade_count):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # VALIDATION MDD
    column_index += 1
    for index, row in enumerate(validation_df_mdd):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # VALIDATION ROMAD
    column_index += 1
    for index, row in enumerate(validation_df_romad):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)
    
    column_index += 1
    # USO IL CICLO DELLE SOGLIE PER PRINTARE LOS TESSO VALORE SU PIU' RIGHE DI BH
    for index, thr in enumerate(thrs): 
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), validation_bh_df_return, cell_format_2)
        worksheet.write(excel_column_list[column_index + 1] + str(row_index + index + 3), validation_bh_df_mdd, cell_format_2)
        worksheet.write(excel_column_list[column_index + 2] + str(row_index + index + 3), validation_bh_df_romad, cell_format_2)

     # TEST RETURN
    column_index += 3
    for index, row in enumerate(test_df):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # TEST ACCURACY
    column_index += 1
    for index, row in enumerate(test_df_accuracy):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)
    
    # TEST COVERAGE
    column_index += 1
    for index, row in enumerate(test_df_coverage):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # TEST TRADE COUNT
    column_index += 1
    for index, row in enumerate(test_df_trade_count):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # TEST MDD
    column_index += 1
    for index, row in enumerate(test_df_mdd):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    # TEST ROMAD
    column_index += 1
    for index, row in enumerate(test_df_romad):
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), row, cell_format_2)

    column_index += 1
    # USO IL CICLO DELLE SOGLIE PER PRINTARE LOS TESSO VALORE SU PIU' RIGHE DI BH
    for index, thr in enumerate(thrs): 
        worksheet.write(excel_column_list[column_index] + str(row_index + index + 3), test_bh_df_return, cell_format_2)
        worksheet.write(excel_column_list[column_index + 1] + str(row_index + index + 3), test_bh_df_mdd, cell_format_2)
        worksheet.write(excel_column_list[column_index + 2] + str(row_index + index + 3), test_bh_df_romad, cell_format_2)
##################################################################

excel_column_list = gen_list('ZZZ')

#experiments =  ['delta_experiment_next_period_1', 
#                'delta_experiment_next_period_2', 
#                'delta_experiment_next_period_3', 
#                'delta_experiment_next_period_4', 
#                'delta_experiment_next_period_5']

#experiments = [
#                '3_months_delta_experiment_1', 
#                '3_months_delta_experiment_2',
#                '3_months_delta_experiment_3',
#                '3_months_delta_experiment_4',
#                '3_months_delta_experiment_5']


experiments = ['multi_company_exp_5']
experiments = ['multi_company_exp_7_small']

initial_validation_row_index = 1
initial_test_row_index = 1

# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('../excel/scheda_grouped_experiment_multi_companies.xlsx')

worksheet = workbook.add_worksheet('Report by walk')

row_index = 1
column_index = 1

# SCHEDA WALK PER WALK
for run_index, experiment in enumerate(experiments):
    
    validation_df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/' + experiment + '/results/ensemble/validation/validation_all_thr_results.csv')
    validation_results_handler = ResultsHandlerExcel(experiment_name=experiment, dataset='sp500')
    validation_df_accuracy, validation_df_coverage, validation_df_trade_count, validation_df_mdd, validation_df_romad = validation_results_handler.generate_ensemble_and_plots(set_type='validation')
    
    test_df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/' + experiment + '/results/ensemble/test/test_all_thr_results.csv')
    test_results_handler = ResultsHandlerExcel(experiment_name=experiment, dataset='sp500')
    test_df_accuracy, test_df_coverage, test_df_trade_count, test_df_mdd, test_df_romad = test_results_handler.generate_ensemble_and_plots(set_type='test')
    
    # BH
    validation_bh_df_return, validation_bh_df_mdd, validation_bh_df_romad, validation_bh_return_totale = validation_results_handler.generate_buy_hold(set_type='validation')
    test_bh_df_return, test_bh_df_mdd, test_bh_df_romad, test_bh_return_totale = test_results_handler.generate_buy_hold(set_type='test')

    headers = validation_df.columns.values
    
    headers = np.delete(headers, 0) # ELIMINO LA COLONNA 0 OVVERO THR
    headers = np.delete(headers, -1) # ELIMINO LA COLONNA 0 OVVERO WALK_ALL
    
    # CICLO PER OGNI COLONNA, OVVERO UN WALK
    for index_header, header in enumerate(headers):
        gen_header_table(workbook=workbook, 
                        worksheet=worksheet, 
                        row_index=row_index, 
                        column_index=column_index, 
                        run_index=run_index + 1,
                        walk_index=index_header + 1)
        
        
        
        fill_return_table(  workbook=workbook, 
                            worksheet=worksheet,
                            # VALIDATION 
                            validation_df=validation_df[header], 
                            validation_df_accuracy=validation_df_accuracy[header],
                            validation_df_coverage=validation_df_coverage[header],
                            validation_df_trade_count=validation_df_trade_count[header],
                            validation_df_mdd=validation_df_mdd[header],
                            validation_df_romad=validation_df_romad[header],
                            #TEST
                            test_df=test_df[header],
                            test_df_accuracy=test_df_accuracy[header], 
                            test_df_coverage=test_df_coverage[header],
                            test_df_trade_count=test_df_trade_count[header], 
                            test_df_mdd=test_df_mdd[header],
                            test_df_romad=test_df_romad[header],
                            #BH
                            validation_bh_df_return=validation_bh_df_return[header],
                            validation_bh_df_romad=validation_bh_df_romad[header],
                            validation_bh_df_mdd=validation_bh_df_mdd[header],
                            test_bh_df_return=test_bh_df_return[header],
                            test_bh_df_romad=test_bh_df_romad[header],
                            test_bh_df_mdd=test_bh_df_mdd[header],
                            # ROW COLUMN
                            row_index=row_index, 
                            column_index=column_index)
        
        column_index = column_index + 19

    column_index = 1
    row_index = row_index + 10


#############################
# Secondo foglio con dati globali 
#############################

worksheet_2 = workbook.add_worksheet('Report global')
row_index = 1
column_index = 1

# SCHEDA WALK ALL
for run_index, experiment in enumerate(experiments):
    #validation
    validation_df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/' + experiment + '/results/ensemble/validation/validation_all_thr_results.csv')
    validation_results_handler = ResultsHandlerExcel(experiment_name=experiment, dataset='sp500')
    validation_df_accuracy, validation_df_coverage, validation_df_trade_count, validation_df_mdd, validation_df_romad = validation_results_handler.generate_ensemble_and_plots_global(set_type='validation')

    #test
    test_df = pd.read_csv('C:/Users/andre/Documents/GitHub/PhD-Market-Nets/experiments/' + experiment + '/results/ensemble/test/test_all_thr_results.csv')
    test_results_handler = ResultsHandlerExcel(experiment_name=experiment, dataset='sp500')
    test_df_accuracy, test_df_coverage, test_df_trade_count, test_df_mdd, test_df_romad = test_results_handler.generate_ensemble_and_plots_global(set_type='test')
    
    # BH
    validation_bh_df_return, validation_bh_df_mdd, validation_bh_df_romad, validation_bh_return_totale = validation_results_handler.generate_buy_hold_global(set_type='validation')
    test_bh_df_return, test_bh_df_mdd, test_bh_df_romad, test_bh_return_totale= test_results_handler.generate_buy_hold_global(set_type='test')

    headers = ['walk_all']
    
    #headers = np.delete(headers, 0) # ELIMINO LA COLONNA 0 OVVERO THR
    #headers = np.delete(headers, -1) # ELIMINO LA COLONNA 0 OVVERO WALK_ALL

    gen_header_table(workbook=workbook, 
                        worksheet=worksheet_2, 
                        row_index=row_index, 
                        column_index=column_index, 
                        run_index=run_index + 1,
                        walk_index=1)

    '''
    # CICLO PER OGNI COLONNA, OVVERO UN WALK
    media_validation_accuracy = []
    media_validation_coverage = []
    somma_validation_trade_count = []

    media_test_accuracy = []
    media_test_coverage = []
    somma_test_trade_count = []

    thrs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # ciclo per ogni soglia (riga)
    for index_thr, thr in enumerate(thrs):
        val_somma_accuracy = 0
        val_somma_coverage = 0
        val_somma_trade_count = 0

        test_somma_accuracy = 0
        test_somma_coverage = 0
        test_somma_trade_count = 0
        # CICLO PER OGNI COLONNA, OVVERO UN WALK (colonna)
        for index_header, header in enumerate(headers):
            # validation
            val_somma_accuracy += validation_df_accuracy.iloc[[index_thr]][header]
            val_somma_coverage += validation_df_coverage.iloc[[index_thr]][header]
            val_somma_trade_count += validation_df_trade_count.iloc[[index_thr]][header]

            # validation
            test_somma_accuracy += test_df_accuracy.iloc[[index_thr]][header]
            test_somma_coverage += test_df_coverage.iloc[[index_thr]][header]
            test_somma_trade_count += test_df_trade_count.iloc[[index_thr]][header]
        
        #uscito dal ciclo effettuo le divisioni quando serve
        media_validation_accuracy.append(val_somma_accuracy / len(headers))
        media_validation_coverage.append(val_somma_coverage / len(headers))
        somma_validation_trade_count.append(val_somma_trade_count)

        media_test_accuracy.append(test_somma_accuracy / len(headers))
        media_test_coverage.append(test_somma_coverage / len(headers))
        somma_test_trade_count.append(test_somma_trade_count)

    # IN QUESTO MODO SO QUANTE RIGHE CI SONO 
    numero_righe = validation_bh_df_return.shape[0]
    numero_colonne = validation_bh_df_return.shape[1] - 1
    #print(validation_bh_df_return)
    '''
    print(validation_bh_df_return)
    for index_header, header in enumerate(headers):
        fill_return_table_global(  workbook=workbook, 
                                worksheet=worksheet_2,
                                # VALIDATION 
                                validation_df=validation_df[header], 
                                validation_df_accuracy=validation_df_accuracy[header],
                                validation_df_coverage=validation_df_coverage[header],
                                validation_df_trade_count=validation_df_trade_count[header],
                                validation_df_mdd=validation_df_mdd[header],
                                validation_df_romad=validation_df_romad[header],
                                #TEST
                                test_df=test_df[header],
                                test_df_accuracy=test_df_accuracy[header], 
                                test_df_coverage=test_df_coverage[header],
                                test_df_trade_count=test_df_trade_count[header], 
                                test_df_mdd=test_df_mdd[header],
                                test_df_romad=test_df_romad[header],
                                #BH
                                validation_bh_df_return=validation_bh_df_return[header],
                                validation_bh_df_romad=validation_bh_df_romad[header],
                                validation_bh_df_mdd=validation_bh_df_mdd[header],
                                test_bh_df_return=test_bh_df_return[header],
                                test_bh_df_mdd=test_bh_df_mdd[header],
                                test_bh_df_romad=test_bh_df_romad[header],
                                
                                # ROW COLUMN
                                row_index=row_index, 
                                column_index=column_index)
                                

    
workbook.close()     

