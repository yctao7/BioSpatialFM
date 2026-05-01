# Extract CODEX patches
python tutorials/utils/patch_extractor.py \
    --input_dir /nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/hpapdata \
    --output_dir /scratch/drjieliu_owned_root/drjieliu_owned1/peterszj/patches/codex

# Extract IMC patches (uncomment if needed)
# python tutorials/utils/patch_extractor.py \
#     --input_dir /nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/IMC/hpapdata \
#     --output_dir /scratch/drjieliu_owned_root/drjieliu_owned1/peterszj/patches/imc \
#     --modality imc