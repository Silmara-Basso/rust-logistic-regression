// Deploy de Modelo de Classificação Através de API em Linguagem Rust

// Importa os módulos necessários do linfa para pré-processamento e treino
use linfa::prelude::*;

// Importa o módulo de regressão logística do linfa_logistic
use linfa_logistic::LogisticRegression;

// Importa módulos do ndarray para trabalhar com arrays numéricos
use ndarray::{Array2, Array1};

// Importa Rocket para criar a API, incluindo macros e tipos JSON
use rocket::{post, serde::json::Json, serde::Deserialize, serde::Serialize, routes};

// Importa módulos para manipular arquivos e entrada/saída
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter};

// Importa funções de serialização e desserialização da bincode
use bincode::{serialize_into, deserialize_from};

// Importa o ReaderBuilder para trabalhar com CSV
use csv::ReaderBuilder;

// Define a estrutura para receber os novos dados de entrada via JSON
#[derive(Deserialize)]
struct PredictRequest {

    // Vetor de atributos fornecidos pelo usuário
    features: Vec<f64>, 
}

// Define a estrutura para enviar a classe prevista como resposta
#[derive(Serialize)]
struct PredictResponse {

    // A classe prevista pelo modelo
    class: usize, 
}

// Função para carregar os dados de um arquivo CSV para treinar o modelo
fn carrega_csv(filename: &str) -> (Array2<f64>, Array1<usize>) {
    
    // Cria um leitor de CSV e garante que ele tenha cabeçalhos
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(filename).expect("Falha ao abrir o CSV");
    let mut features_vec = Vec::new();
    let mut targets_vec = Vec::new();

    // Para cada linha do CSV, extrai as características e o alvo
    for result in rdr.records() {
        let record = result.expect("Falha ao ler linha do CSV");
        let mut row_features = Vec::new();

        // Separa as características do alvo (última coluna)
        for (i, field) in record.iter().enumerate() {
            if i == record.len() - 1 {
                // Converte o alvo para usize
                targets_vec.push(field.parse::<usize>().expect("Falha ao converter target"));
            } else {
                // Converte as características para f64
                row_features.push(field.parse::<f64>().expect("Falha ao converter feature"));
            }
        }
        features_vec.push(row_features);
    }

    // Converte os vetores de características e alvos em arrays do ndarray
    let n_samples = features_vec.len();
    let n_features = features_vec[0].len();
    let features = Array2::from_shape_vec((n_samples, n_features), features_vec.into_iter().flatten().collect()).unwrap();
    let targets = Array1::from_vec(targets_vec);

    // Retorna as características (atributos) e alvos
    (features, targets) 
}

// Função para treinar o modelo e salvá-lo
fn treina_salva_modelo() {
    
    // Carrega os dados do arquivo CSV
    let (features, targets) = carrega_csv("dados.csv");

    // Cria o dataset para treinamento
    let dataset = Dataset::new(features, targets); 

    // Treina o modelo de regressão logística
    let fitted_model = LogisticRegression::default().fit(&dataset).unwrap();

    // Cria um arquivo para salvar o modelo treinado
    let file = File::create("modelo.bin").expect("Falha ao criar arquivo para salvar o modelo");
    let mut writer = BufWriter::new(file);
    
    // Serializa o modelo treinado e o salva no arquivo
    serialize_into(&mut writer, &fitted_model).expect("Falha ao salvar o modelo no arquivo");
}

// Função para carregar o modelo treinado de um arquivo
fn carrega_modelo() -> Result<linfa_logistic::FittedLogisticRegression<f64, usize>, String> {
    
    // Abre o arquivo onde o modelo treinado foi salvo
    let file = OpenOptions::new().read(true).open("modelo.bin").map_err(|_| "Falha ao abrir o arquivo do modelo".to_string())?;
    let reader = BufReader::new(file);
    
    // Desserializa o modelo a partir do arquivo
    let fitted_model: linfa_logistic::FittedLogisticRegression<f64, usize> = deserialize_from(reader).map_err(|e| format!("Falha ao carregar o modelo: {}", e))?;
    
    // Retorna o modelo carregado
    Ok(fitted_model) 
}

// Função que define a API com o endpoint de previsão
#[post("/predict", format = "json", data = "<input>")]
fn predict(input: Json<PredictRequest>) -> Json<PredictResponse> {
    
    // Tenta carregar o modelo treinado
    match carrega_modelo() {
        
        Ok(fitted_model) => {

            // Converte as características recebidas em um array 2D para a previsão
            let features_array: Array2<f64> = Array2::from_shape_vec((1, input.features.len()), input.features.clone()).unwrap();
            
            // Faz a previsão usando o modelo carregado
            let prediction = fitted_model.predict(&features_array);
            
            // Retorna a classe prevista como resposta em JSON
            Json(PredictResponse { class: prediction[0] })
        },
        Err(e) => {
            // Em caso de erro ao carregar o modelo, exibe mensagem de erro
            eprintln!("Erro ao carregar o modelo: {}", e);
            panic!("Falha ao carregar o modelo!");
        }
    }
}

// Função principal que inicia o servidor e monta a rota para o endpoint de previsão
#[rocket::main]
async fn main() {

    // Treina e salva o modelo ao iniciar o servidor
    treina_salva_modelo();

    rocket::build()

        // Define as rotas disponíveis na API
        .mount("/", routes![predict]) 
        .launch()
        .await

        // Inicia o servidor Rocket
        .unwrap(); 
}



