#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include "esp_timer.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

// ============================
// MFCC INPUTS
// ============================
#define MODEL_INPUT_NAME mfcc_input_left_1
#include "mfcc_input_left_1.h"
#undef MODEL_INPUT_NAME

#define MODEL_INPUT_NAME mfcc_input_left_2
#include "mfcc_input_left_2.h"
#undef MODEL_INPUT_NAME

#define MODEL_INPUT_NAME mfcc_input_yes_1
#include "mfcc_input_yes_1.h"
#undef MODEL_INPUT_NAME

#define MODEL_INPUT_NAME mfcc_input_yes_2
#include "mfcc_input_yes_2.h"
#undef MODEL_INPUT_NAME

#define MODEL_INPUT_NAME mfcc_input_down_1
#include "mfcc_input_down_1.h"
#undef MODEL_INPUT_NAME

#define MODEL_INPUT_NAME mfcc_input_down_2
#include "mfcc_input_down_2.h"
#undef MODEL_INPUT_NAME

#define MODEL_INPUT_NAME mfcc_input_up_1
#include "mfcc_input_up_1.h"
#undef MODEL_INPUT_NAME

#define MODEL_INPUT_NAME mfcc_input_up_2
#include "mfcc_input_up_2.h"
#undef MODEL_INPUT_NAME

// ============================
// Tensor Arena
// ============================
constexpr int kTensorArenaSize = 100 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// ============================
// Test structure
// ============================
struct TestCase {
    const char* name;
    const int8_t* data;
};

static const TestCase tests[] = {
    {"left_1", mfcc_input_left_1},
    {"left_2", mfcc_input_left_2},
    {"yes_1", mfcc_input_yes_1},
    {"yes_2", mfcc_input_yes_2},
    {"down_1", mfcc_input_down_1},
    {"down_2", mfcc_input_down_2},
    {"up_1", mfcc_input_up_1},
    {"up_2", mfcc_input_up_2},
};

static const int num_tests = sizeof(tests) / sizeof(tests[0]);

// ============================
// Argmax helper
// ============================
int argmax(const int8_t* data, int size) {
    int best_i = 0;
    int best_v = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    return best_i;
}

// ============================
// Labels (match your model)
// ============================
const char* labels[] = {
    "down", "go", "left", "no", "off", "on",
    "right", "silence", "stop", "unknown",
    "up", "yes"
};

// ============================
// Main
// ============================
extern "C" void app_main(void)
{
    printf("\n===== TFLM KWS START =====\n");

    const tflite::Model* model = tflite::GetModel(model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("❌ Model schema mismatch!\n");
        return;
    }

    // Ops resolver
    static tflite::MicroMutableOpResolver<8> resolver;

    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMean();
    resolver.AddPad();

    // Interpreter
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("❌ Tensor allocation failed\n");
        return;
    }

    printf("✅ Arena used: %d bytes\n", interpreter.arena_used_bytes());

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf("Input shape: ");
    for (int i = 0; i < input->dims->size; i++) {
        printf("%d ", input->dims->data[i]);
    }
    printf("\n");

    printf("Input scale: %f zero_point: %" PRId32 "\n",
           input->params.scale,
           input->params.zero_point);

    // ============================
    // Run tests
    // ============================
    int correct = 0;

    for (int i = 0; i < num_tests; i++) {

        printf("\n-------------------------\n");
        printf("Test: %s\n", tests[i].name);

        memcpy(input->data.int8, tests[i].data, input->bytes);

        int64_t t0 = esp_timer_get_time();
        TfLiteStatus status = interpreter.Invoke();
        int64_t t1 = esp_timer_get_time();

        if (status != kTfLiteOk) {
            printf("❌ Inference failed\n");
            continue;
        }

        int out_size = output->bytes;

        int pred = argmax(output->data.int8, out_size);

        printf("Prediction: %s\n", labels[pred]);
        printf("Time: %lld us\n", (long long)(t1 - t0));

        printf("Logits (top 5): ");

        // print top values safely
        for (int k = 0; k < 5 && k < out_size; k++) {
            printf("%d ", output->data.int8[k]);
        }
        printf("\n");

        // simple accuracy check (based on filename)
        if (strstr(tests[i].name, labels[pred]) != NULL) {
            correct++;
        }
    }

    printf("\n=========================\n");
    printf("Accuracy: %d/%d\n", correct, num_tests);
    printf("=========================\n");

    printf("===== DONE =====\n");
}