A fast BPE tokenizer written in Rust.

## Fast on small inputs

After pre-tokenization splitting, most inputs will be very small. FastBPE is absurdly fast on small inputs.

![Screenshot 2024-09-05 at 8 01 24 PM](https://github.com/user-attachments/assets/cb8ee307-dafb-4199-acdd-3495e7c3e8d0)

## Fast on giant inputs

Even if you don't pre-tokenize, FastBPE is takes linear time for any input size. This makes it very fast on giant inputs.

![Screenshot 2024-09-06 at 6 59 52 AM](https://github.com/user-attachments/assets/1120bce3-ad53-4037-adb6-f7c1f602ce1e)
