# lay.ps1 - Script PowerShell para treinar todas as layers com base no main.py e arquivos YAML

$scriptPath = "F:\Git\CAL-MOS\src\main.py"
$configDir = "F:\Git\CAL-MOS\config\embeddings\geral\all_layers\wav2vec2-large"

# Verifica se os caminhos existem
if (!(Test-Path $scriptPath)) {
    Write-Host "Erro: O script principal não foi encontrado em: $scriptPath"
    exit 1
}
if (!(Test-Path $configDir)) {
    Write-Host "Erro: O diretório de configurações não foi encontrado em: $configDir"
    exit 1
}

# Define o interpretador Python
$pythonExe = "python"

# Percorre todas as layers de 0 a 25
for ($layer = 8; $layer -le 15; $layer++) {
    $configFilename = "config_w2v2_large_layer-$layer-seqaug.yaml"
    $configPath = Join-Path $configDir $configFilename

    Write-Host "[$layer] Iniciando treino da layer $layer"
    Write-Host "       Configuração: $configPath"

    if (!(Test-Path $configPath)) {
        Write-Host "Erro: Arquivo de configuração não encontrado: $configPath"
        exit 1
    }

    # Executa o main.py com o argumento da config, usando encoding UTF-8 e variável de ambiente ajustada
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"

    $process = Start-Process -FilePath $pythonExe `
                             -ArgumentList "`"$scriptPath`" -c `"$configPath`"" `
                             -NoNewWindow -Wait -PassThru

    if ($process.ExitCode -ne 0) {
        Write-Host "Erro ao treinar a layer $layer. Código de saída: $($process.ExitCode)"
        exit $process.ExitCode
    }

    Write-Host "Layer $layer finalizada com sucesso."
}

Write-Host "Todos os treinamentos de layers foram concluídos com sucesso."
