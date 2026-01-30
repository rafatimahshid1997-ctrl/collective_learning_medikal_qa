"""
Collective Learning for Turkish Medical Question Answering
Project #9: Self-Improving Language Models
Model: cosmos:T1-2B
Dataset: hoatac/Medikal-QA-Turkish
"""

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import gc

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    """Configuration for the collective learning experiment"""
    
    # Model configuration
    MODEL_NAME = "cosmocanyon/cosmos-T1-2B"  # Turkish-focused model
    
    # Dataset configuration
    DATASET_NAME = "hoatac/Medikal-QA-Turkish"
    TRAIN_SIZE_PER_ITER = 500
    TEST_SIZE = 500
    NUM_ITERATIONS = 3
    
    # LoRA configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Training configuration
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 512
    
    # Paths
    OUTPUT_DIR = "./collective_learning_outputs"
    RESULTS_FILE = "experimental_results.json"
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# PROMPTS FOR EVALUATION AND IMPROVEMENT
# ==========================================

class PromptTemplates:
    """High-quality prompts for self-evaluation and improvement"""
    
    @staticmethod
    def get_answer_prompt(question: str) -> str:
        """Generate answer to medical question"""
        return f"""Aşağıdaki tıbbi soruyu yanıtlayın. Yanıtınız bilimsel olarak doğru, güvenli ve anlaşılır olmalıdır.

Soru: {question}

Yanıt:"""
    
    @staticmethod
    def get_evaluation_prompt(question: str, answer: str) -> str:
        """Evaluate the quality of a medical answer"""
        return f"""Aşağıdaki tıbbi soru-cevap çiftini değerlendirin ve JSON formatında puanlayın.

Soru: {question}

Yanıt: {answer}

Değerlendirme kriterleri:
1. Tıbbi Doğruluk (0-10): Yanıt bilimsel olarak ne kadar doğru?
2. Güvenlik (0-10): Yanıt hastalar için güvenli mi? Tehlikeli tavsiyeler var mı?
3. Netlik (0-10): Yanıt ne kadar anlaşılır ve net?
4. Halüsinasyon (Evet/Hayır): Yanıt gerçek olmayan bilgiler içeriyor mu?
5. Eksik Bilgi (Evet/Hayır): Kritik bilgiler eksik mi?
6. Belirsizlik İfadesi (Evet/Hayır): Gerektiğinde "kesin değilim" gibi ifadeler kullanılmış mı?

Çıktıyı şu formatta verin:
{{
  "tikbi_dogruluk": <0-10>,
  "guvenlik": <0-10>,
  "netlik": <0-10>,
  "halusinasyon": <"Evet" veya "Hayır">,
  "eksik_bilgi": <"Evet" veya "Hayır">,
  "belirsizlik_ifadesi": <"Evet" veya "Hayır">,
  "genel_puan": <0-10>,
  "yorumlar": "<kısa değerlendirme>"
}}

Değerlendirme:"""
    
    @staticmethod
    def get_improvement_prompt(question: str, answer: str, evaluation: dict) -> str:
        """Improve answer based on evaluation"""
        eval_summary = f"""
Mevcut Yanıt Değerlendirmesi:
- Tıbbi Doğruluk: {evaluation.get('tikbi_dogruluk', 'N/A')}/10
- Güvenlik: {evaluation.get('guvenlik', 'N/A')}/10
- Netlik: {evaluation.get('netlik', 'N/A')}/10
- Halüsinasyon: {evaluation.get('halusinasyon', 'N/A')}
- Eksik Bilgi: {evaluation.get('eksik_bilgi', 'N/A')}
- Yorumlar: {evaluation.get('yorumlar', 'N/A')}
"""
        
        return f"""Aşağıdaki tıbbi sorunun yanıtını değerlendirme sonuçlarına göre iyileştirin.

Soru: {question}

Mevcut Yanıt: {answer}

{eval_summary}

İyileştirme kuralları:
1. Tıbbi doğruluğu artırın
2. Güvensiz tavsiyeleri düzeltin
3. Eksik bilgileri ekleyin
4. Halüsinasyonları kaldırın
5. Gerektiğinde belirsizlik ifadesi kullanın (örn: "Bu durum için mutlaka bir doktora danışmalısınız")
6. Daha net ve anlaşılır yazın

İyileştirilmiş Yanıt:"""


# ==========================================
# DATA MANAGEMENT
# ==========================================

class DataManager:
    """Manages dataset loading, splitting, and preparation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
        self.train_questions = []
        self.test_questions = []
        
    def load_dataset(self):
        """Load the Turkish medical QA dataset"""
        print("Loading dataset...")
        self.dataset = load_dataset(self.config.DATASET_NAME)
        
        # Extract questions from the dataset
        if 'train' in self.dataset:
            data = self.dataset['train']
        else:
            data = self.dataset[list(self.dataset.keys())[0]]
        
        # Assuming dataset has 'question' and 'answer' fields
        all_questions = []
        for item in data:
            if 'question' in item:
                all_questions.append({
                    'question': item['question'],
                    'original_answer': item.get('answer', '')
                })
        
        # Split into train and test
        np.random.seed(42)
        indices = np.random.permutation(len(all_questions))
        
        total_train = self.config.TRAIN_SIZE_PER_ITER * self.config.NUM_ITERATIONS
        train_indices = indices[:total_train]
        test_indices = indices[total_train:total_train + self.config.TEST_SIZE]
        
        self.train_questions = [all_questions[i] for i in train_indices]
        self.test_questions = [all_questions[i] for i in test_indices]
        
        print(f"Loaded {len(self.train_questions)} training questions")
        print(f"Loaded {len(self.test_questions)} test questions")
        
    def get_iteration_data(self, iteration: int):
        """Get training data for specific iteration"""
        start_idx = iteration * self.config.TRAIN_SIZE_PER_ITER
        end_idx = start_idx + self.config.TRAIN_SIZE_PER_ITER
        return self.train_questions[start_idx:end_idx]


# ==========================================
# MODEL MANAGER
# ==========================================

class ModelManager:
    """Manages model loading, generation, and LoRA training"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.base_model = None
        
    def load_model(self):
        """Load the base model and tokenizer"""
        print(f"Loading model: {self.config.MODEL_NAME}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            trust_remote_code=True
        )
        
        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.base_model = self.model
        print("Model loaded successfully")
        
    def generate_text(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        if prompt in generated_text:
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def apply_lora(self):
        """Apply LoRA configuration to model"""
        print("Applying LoRA configuration...")
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.LORA_TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def train_lora(self, training_data: list, iteration: int):
        """Train model with LoRA on improved answers"""
        print(f"\nTraining iteration {iteration}...")
        
        # Prepare training dataset
        train_texts = []
        for item in training_data:
            text = f"Soru: {item['question']}\n\nYanıt: {item['improved_answer']}"
            train_texts.append(text)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples,
                truncation=True,
                max_length=self.config.MAX_LENGTH,
                padding="max_length"
            )
        
        tokenized_data = [tokenize_function(text) for text in train_texts]
        
        # Create dataset
        class QADataset(torch.utils.data.Dataset):
            def __init__(self, tokenized_data):
                self.data = tokenized_data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.tensor(self.data[idx]['input_ids']),
                    'attention_mask': torch.tensor(self.data[idx]['attention_mask']),
                    'labels': torch.tensor(self.data[idx]['input_ids'])
                }
        
        train_dataset = QADataset(tokenized_data)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config.OUTPUT_DIR}/iteration_{iteration}",
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=self.config.LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_steps=50,
            report_to="none"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train
        trainer.train()
        
        # Clear memory
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Training iteration {iteration} completed")


# ==========================================
# COLLECTIVE LEARNING ENGINE
# ==========================================

class CollectiveLearningEngine:
    """Main engine for the collective learning loop"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.model_manager = ModelManager(config)
        self.prompt_templates = PromptTemplates()
        self.results = {
            'iterations': [],
            'test_results': []
        }
        
    def initialize(self):
        """Initialize the system"""
        print("="*60)
        print("COLLECTIVE LEARNING SYSTEM INITIALIZATION")
        print("="*60)
        
        self.data_manager.load_dataset()
        self.model_manager.load_model()
        
        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
    def evaluate_answer(self, question: str, answer: str) -> dict:
        """Evaluate an answer using the model"""
        eval_prompt = self.prompt_templates.get_evaluation_prompt(question, answer)
        eval_text = self.model_manager.generate_text(eval_prompt, max_new_tokens=300)
        
        # Parse JSON from evaluation
        try:
            # Extract JSON from response
            if '{' in eval_text and '}' in eval_text:
                json_start = eval_text.find('{')
                json_end = eval_text.rfind('}') + 1
                json_str = eval_text[json_start:json_end]
                evaluation = json.loads(json_str)
            else:
                # Default evaluation if parsing fails
                evaluation = {
                    'tikbi_dogruluk': 5,
                    'guvenlik': 5,
                    'netlik': 5,
                    'halusinasyon': 'Hayır',
                    'eksik_bilgi': 'Evet',
                    'belirsizlik_ifadesi': 'Hayır',
                    'genel_puan': 5,
                    'yorumlar': 'Otomatik değerlendirme'
                }
        except:
            evaluation = {
                'tikbi_dogruluk': 5,
                'guvenlik': 5,
                'netlik': 5,
                'halusinasyon': 'Hayır',
                'eksik_bilgi': 'Evet',
                'belirsizlik_ifadesi': 'Hayır',
                'genel_puan': 5,
                'yorumlar': 'Değerlendirme hatası'
            }
        
        return evaluation
    
    def improve_answer(self, question: str, answer: str, evaluation: dict) -> str:
        """Improve answer based on evaluation"""
        improve_prompt = self.prompt_templates.get_improvement_prompt(
            question, answer, evaluation
        )
        improved_answer = self.model_manager.generate_text(improve_prompt, max_new_tokens=300)
        return improved_answer
    
    def run_iteration(self, iteration: int):
        """Run one iteration of collective learning"""
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        
        # Get training data for this iteration
        iteration_data = self.data_manager.get_iteration_data(iteration)
        
        # Step 1: Generate initial answers
        print(f"Step 1: Generating initial answers for {len(iteration_data)} questions...")
        for item in tqdm(iteration_data):
            answer_prompt = self.prompt_templates.get_answer_prompt(item['question'])
            item['initial_answer'] = self.model_manager.generate_text(answer_prompt)
        
        # Step 2: Evaluate answers
        print("Step 2: Evaluating answers...")
        for item in tqdm(iteration_data):
            item['evaluation'] = self.evaluate_answer(
                item['question'],
                item['initial_answer']
            )
        
        # Step 3: Improve answers
        print("Step 3: Improving answers based on evaluation...")
        for item in tqdm(iteration_data):
            item['improved_answer'] = self.improve_answer(
                item['question'],
                item['initial_answer'],
                item['evaluation']
            )
        
        # Calculate iteration statistics
        avg_score = np.mean([item['evaluation'].get('genel_puan', 5) 
                            for item in iteration_data])
        
        print(f"\nIteration {iteration} Statistics:")
        print(f"  Average Initial Score: {avg_score:.2f}/10")
        
        # Step 4: Train on improved answers
        if iteration < self.config.NUM_ITERATIONS - 1:  # Don't train after last iteration
            self.model_manager.apply_lora()
            self.model_manager.train_lora(iteration_data, iteration)
        
        # Store results
        self.results['iterations'].append({
            'iteration': iteration,
            'avg_score': float(avg_score),
            'num_samples': len(iteration_data)
        })
        
        # Save iteration data
        iteration_file = f"{self.config.OUTPUT_DIR}/iteration_{iteration}_data.json"
        with open(iteration_file, 'w', encoding='utf-8') as f:
            json.dump(iteration_data, f, ensure_ascii=False, indent=2)
        
        return iteration_data
    
    def evaluate_on_test_set(self):
        """Evaluate model on fixed test set after all iterations"""
        print(f"\n{'='*60}")
        print("FINAL EVALUATION ON TEST SET")
        print(f"{'='*60}")
        
        test_results = []
        
        for item in tqdm(self.data_manager.test_questions[:50]):  # Sample for memory
            answer_prompt = self.prompt_templates.get_answer_prompt(item['question'])
            answer = self.model_manager.generate_text(answer_prompt)
            evaluation = self.evaluate_answer(item['question'], answer)
            
            test_results.append({
                'question': item['question'],
                'answer': answer,
                'evaluation': evaluation
            })
        
        avg_test_score = np.mean([r['evaluation'].get('genel_puan', 5) 
                                  for r in test_results])
        
        print(f"\nTest Set Average Score: {avg_test_score:.2f}/10")
        
        self.results['test_results'] = test_results
        self.results['avg_test_score'] = float(avg_test_score)
        
    def save_results(self):
        """Save all experimental results"""
        results_path = f"{self.config.OUTPUT_DIR}/{self.config.RESULTS_FILE}"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {results_path}")
    
    def run(self):
        """Run the complete collective learning experiment"""
        self.initialize()
        
        # Run iterations
        for iteration in range(self.config.NUM_ITERATIONS):
            self.run_iteration(iteration)
        
        # Final evaluation
        self.evaluate_on_test_set()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*60)


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    
    # Create and run collective learning engine
    engine = CollectiveLearningEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
