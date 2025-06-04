from .nodes import *

NODE_CLASS_MAPPINGS = {
    ## Models
    "Load UNet Model with Name JNK":LoadModelName_Diff_JNK,
    "Load Checkpoint Model with Name JNK":LoadModelName_Chpt_JNK,
    "Load LoRA with Name JNK":LoadLoRAName_JNK,
    ## Video
    "Save Frame JNK":SaveFrame_JNK,
    "Save Video Images JNK":SaveVideoImages_JNK,
    ## Image
    "Save Static Image JNK":SaveStaticImage_JNK,
    "Load Image if Exist JNK":LoadImageWithCheck_JNK,
    "Image Filter Loader JNK":ImageFilterLoader_JNK,
    "Stroke RGBA Image JNK":StrokeImage_JNK,
    "Create RGBA Image JNK":AlphaImageNode_JNK,
    "Add Layer Overlay JNK":AddLayerOverlay_JNK,
    "Get One Alpha Layer JNK":GetAlphaLayers_JNK,
    "Get All Alpha Layers JNK":GetAllAlphaLayers_JNK,
    ## Upscale
    'Topaz Photo Upscaler (Autopilot) JNK':TopazPhotoAI_JNK,
    ## Text
    "Get Text From List by Index JNK":GetTextFromList_JNK,
    "Text Saver JNK":TextSaver_JNK,
    "Get Substring JNK":GetSubstring_JNK,
    "Text to Key JNK":Model2Key_JNK,
    "Text to MD5 JNK":Text2MD5_JNK,
    "Join Strings JNK":JoinStrings_JNK,
    "Get Timestamp JNK":GetTimestamp_JNK,
    ## Logic
    "Switch Integer JNK":SwitchInt_JNK,
    "Switch Index JNK":SwitchIdx_JNK,
    "Get Models JNK":GetModels_JNK,
    ## System
    "Bridge All JNK":BridgeAll_JNK,
    "Queue Stop JNK":QueueStop_JNK,
    "Create Folder JNK":CreateFolder_JNK,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ## Models
    "LoadModelName_Diff_JNK":"Load Diffusion Model with Name",
    "LoadModelName_Unet_JNK":"Load UNet Model with Name",
    "LoadModelName_Chpt_JNK":"Load Checkpoint Model with Name",
    "LoadLoRAName_JNK":"Load LoRA with Name",
    ## Video
    "SaveFrame_JNK":"Save Frame",
    "SaveVideoImages_JNK":"Save Video Images",
    ## Image
    "SaveStaticImage_JNK":"Save Static Image",
    "LoadImageWithCheck_JNK":"Load Image if Exist",
    "ImageFilterLoader_JNK":"Image Filter Loader",
    "StrokeImage_JNK":"Stroke RGBA Image",
    "AlphaImageNode_JNK":"Create RGBA Image",
    "AddLayerOverlay_JNK":"Add Layer Overlay",
    "GetAlphaLayers_JNK":"Get One Alpha Layer",
    "GetAllAlphaLayers_JNK":"Get All Alpha Layers",
    ## Upscale
    'TopazPhotoAI_JNK':'Topaz Photo Upscaler',
    ## Text
    "GetTextFromList_JNK":"Get Text From List by Index",
    "TextSaver_JNK":"Text Save",
    "GetSubstring_JNK":"Get Substring",
    "Model2Key_JNK":"Text to Key",
    "Text2MD5_JNK":"Text to MD5",
    "JoinStrings_JNK":"Join Strings",
    "GetTimestamp_JNK":"Get Timestamp",
    ## Logic
    "SwitchInt_JNK":"Switch (Integer)",
    "SwitchIdx_JNK":"Switch (Index)",
    "GetModels_JNK":"Load Get Models",
    ## System
    "BridgeAll_JNK":"Bridge All",
    "QueueStop_JNK":"Queue Stop",
    "CreateFolder_JNK":"Create Folder",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]