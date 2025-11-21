pub struct ExternalArchiveSource {
    pub dataset: &'static str,
    pub prefix: &'static str,
    pub strip_prefix: &'static str,
    pub remote_dataset: &'static str,
    pub archives: &'static [&'static str],
}

const TREEVGR_ARCHIVES: [&str; 11] = [
    "llava_next_raw_format/llava_next_raw_format_images_1.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_2.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_3.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_4.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_5.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_6.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_7.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_8.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_9.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_10.tar.gz",
    "llava_next_raw_format/llava_next_raw_format_images_11.tar.gz",
];

const EXTERNAL_SOURCES: &[ExternalArchiveSource] = &[ExternalArchiveSource {
    dataset: "HaochenWang/TreeVGR-SFT-35K",
    prefix: "images/",
    strip_prefix: "images/",
    remote_dataset: "lmms-lab/LLaVA-NeXT-Data",
    archives: &TREEVGR_ARCHIVES,
}];

pub fn matching_sources(
    dataset_name: &str,
    reference: &str,
) -> Vec<&'static ExternalArchiveSource> {
    EXTERNAL_SOURCES
        .iter()
        .filter(|source| source.dataset == dataset_name && reference.starts_with(source.prefix))
        .collect()
}
