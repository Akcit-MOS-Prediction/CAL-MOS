import os

# Defina o caminho do arquivo de entrada (que contém filepath,mos)
arquivo_entrada = "/home/pedro-lustosa/Downloads/6572573/main/DATA/sets/test_mos_list.txt"

# Defina o nome do arquivo de saída (que receberá somente as linhas válidas)
arquivo_saida = "/home/pedro-lustosa/Downloads/6572573/main/DATA/sets/edited_test_mos_list.txt"

# Defina a pasta onde estão os arquivos .wav
pasta_wav = "/home/pedro-lustosa/Downloads/6572573/main/DATA/wav"

# Lista para armazenar as linhas que realmente existem
linhas_validas = []

with open(arquivo_entrada, "r", encoding="utf-8") as f:
    # Ler todas as linhas do arquivo
    linhas = f.readlines()

# Vamos manter a primeira linha (cabeçalho) no resultado final, se for o caso
# Ajuste conforme necessário, caso não tenha cabeçalho ou queira descartar.
if len(linhas) > 0:
    # Verificamos se a primeira linha é o cabeçalho:
    if linhas[0].strip().startswith("filepath,mos"):
        # Adicionamos o cabeçalho direto
        linhas_validas.append(linhas[0])
        # Começamos a processar a partir da segunda linha
        linhas_a_processar = linhas[1:]
    else:
        # Caso não haja cabeçalho, processamos tudo
        linhas_a_processar = linhas
else:
    linhas_a_processar = []

# Para cada linha do arquivo (sem o cabeçalho)
for linha in linhas_a_processar:
    linha = linha.strip()
    if not linha:
        continue  # Ignora linhas vazias

    # Dividir a linha em filepath e mos
    partes = linha.split(",")
    if len(partes) != 2:
        # Se a linha não estiver no formato esperado, pula
        continue
    
    filepath, mos = partes
    # Montar o caminho completo do arquivo WAV
    caminho_arquivo_wav = os.path.join(pasta_wav, filepath)

    # Verificar se o arquivo existe
    if os.path.exists(caminho_arquivo_wav):
        # Se existir, adiciona a linha aos válidos
        linhas_validas.append(linha + "\n")

# Escreve as linhas válidas em um novo arquivo
with open(arquivo_saida, "w", encoding="utf-8") as f_out:
    f_out.writelines(linhas_validas)

print(f"Arquivo '{arquivo_saida}' gerado com as linhas cujos áudios existem na pasta '{pasta_wav}'.")
