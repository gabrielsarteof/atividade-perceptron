import subprocess
import webbrowser
import time
import glob
import sys

# Gerar relatório
print("Gerando relatório...")
subprocess.run([sys.executable, "report_generator.py"])

# Encontrar último arquivo
html_files = glob.glob("relatorio_perceptron_*.html")
if html_files:
    latest_file = max(html_files, key=lambda x: x.split('_')[-1])
    
    # Iniciar servidor
    server = subprocess.Popen([sys.executable, "-m", "http.server", "8000"],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    # Abrir diretamente o arquivo
    url = f"http://localhost:8000/{latest_file}"
    webbrowser.open(url)
    print(f"Relatório aberto: {url}")
    
    input("Pressione Enter para parar...")
    server.terminate()