# LaCT ASR Evaluation Report

**Model:** 1024d hidden, 18 layers, 8 LaCT heads
**Training:** Step 91000, Epoch 6
**Date:** 2025-10-18 12:25:20

## Results Summary

| Test Set | WER | CER | Samples | RTF |
|----------|-----|-----|---------|-----|
| dev-clean | 22.67% | 6.69% | 2694 | 0.005x |
| test-clean | 91.93% | 37.42% | 2611 | 0.002x |

## Detailed Results

### dev-clean

- **WER:** 22.67%
- **CER:** 6.69%
- **Samples:** 2694
- **Avg inference time:** 68.0ms
- **Real-Time Factor:** 0.005x
- **Total audio:** 10.03 hours

### test-clean

- **WER:** 91.93%
- **CER:** 37.42%
- **Samples:** 2611
- **Avg inference time:** 28.0ms
- **Real-Time Factor:** 0.002x
- **Total audio:** 10.26 hours

