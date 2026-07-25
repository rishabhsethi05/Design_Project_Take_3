#include "picojpeg.h"

// Define your workload multiplier
#define WORKLOAD_MULTIPLIER 5

// Assuming you have an image array and a callback function defined elsewhere
extern const unsigned char test_jpeg[];
extern unsigned int test_jpeg_len;
extern unsigned int test_jpeg_offset;

// Standard BEEBS entry function
int benchmark(void) {
    int pass;
    uint8_t status = 0;
    pjpeg_image_info_t image_info;

    // Execute the heavy workload 5 times
    for (pass = 0; pass < WORKLOAD_MULTIPLIER; pass++) {

        // 1. Reset the simulated file pointer for each run
        test_jpeg_offset = 0;

        // 2. Initialize the decoder (Parses headers, builds Huffman/Quant tables)
        // This is a highly state-heavy operation
        status = pjpeg_decode_init(&image_info, pjpeg_need_bytes_callback, NULL, 0);

        if (status != 0) {
            return status; // Halt on error
        }

        // 3. The core processing loop
        // This dynamically decodes MCUs (Minimum Coded Units) and causes massive memory turnover
        for ( ; ; ) {
            status = pjpeg_decode_mcu();

            if (status != 0) {
                if (status == PJPG_NO_MORE_BLOCKS) {
                    // Image finished successfully, break to next pass
                    break;
                } else {
                    // An actual error occurred
                    return status;
                }
            }
        }
    }

    // Return 0 indicating all 5 passes completed successfully
    return 0;
}