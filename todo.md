# WIP Notes

## ToDo

- Fix stdout bug in export
- LoRA
- Controlnet
  
## Notes

- SD 2.1 768 required to enable f32 attention in settings
  - Base model seems to be working?!
  
## Questions

- Why does inpainting only have four dims?
  - Apperently model dependent...
- How to overload the refiner model / Why dosent SDXL call the TRT Unet
  - Fixed in dev branch
- What is the num classes in SDXL
  - Don't know. But fair to assume that they are constant

- If the TRT labels are the same as the model can it be selected automatically?

### Optional

- Convert batches in forward call. Iterate over batch.
- Create context without RAM
