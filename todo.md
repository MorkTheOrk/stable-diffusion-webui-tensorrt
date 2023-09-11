# WIP Notes

## ToDo

- Set MinOptMax depending on the model
- Hash for engine params
- Fix stdout bug
- Memory consumption of dynamic shapes
- NaN in SD2.1. with TRT - Enable obey precision?

## Notes

- SD 2.1 required to enable f32 attention in settings
  
## Questions

- Why does inpainting only have four dims?
- How to overload the refiner model
- How to empty U#Net VRAM from Torch
- Why does every encoded sequence have length = 77?