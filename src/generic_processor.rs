use anyhow::Result;
use flate2::read::GzDecoder;
use hf_hub::{
    api::sync::{Api, ApiRepo},
    Repo, RepoType,
};
use polars::prelude::*;
use reqwest::blocking::Client;
use serde_json::{json, Map as JsonMap, Value as JsonValue};
use std::collections::HashSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tar::Archive;

use crate::arrow_reader::dataset_cache_dir;
use crate::dataset_helpers::{
    DatasetItemProcessor, DatasetMetadata, DownloadedFile, ModalitySegment, ModalityType,
    SegmentContent,
};
use crate::polars_dataset_loader::{
    extract_binary_value, extract_list_json, extract_struct_json, extract_text_value,
};
use crate::polars_ext::hash_ops::hash_bytes;
use crate::polars_ext::spatial_ops::{extract_bboxes_from_text, serialize_bboxes};
use crate::reference_sources::{matching_sources, ExternalArchiveSource};
use urlencoding::encode;

/// Generic Polars-based dataset processor that infers modalities from schema metadata.
pub struct GenericDatasetProcessor {
    dataset_name: String,
    http_client: Client,
}

impl GenericDatasetProcessor {
    pub fn new(dataset_name: &str) -> Self {
        Self {
            dataset_name: dataset_name.to_string(),
            http_client: Client::new(),
        }
    }

    /// Convert an entire DataFrame row into modality segments using schema inference.
    pub fn process_row(&self, df: &DataFrame, row_idx: usize) -> Result<Vec<ModalitySegment>> {
        let mut segments = Vec::new();
        let mut seen_hashes = HashSet::new();

        for column_name in df.get_column_names() {
            let series = df.column(column_name)?;
            let modality = infer_modality(column_name, series.dtype());

            match modality {
                ModalityType::Image => {
                    segments.extend(self.extract_image_segments(
                        df,
                        row_idx,
                        column_name,
                        series.dtype(),
                        &mut seen_hashes,
                    )?);
                }
                ModalityType::Audio
                | ModalityType::Video
                | ModalityType::Document
                | ModalityType::Text => {
                    segments.extend(self.extract_textual_segments(
                        df,
                        row_idx,
                        column_name,
                        &modality,
                        series.dtype(),
                    )?);
                }
                ModalityType::Unknown(_) => {
                    // Skip columns that we can't confidently classify
                }
            }
        }

        Ok(segments)
    }

    fn extract_image_segments(
        &self,
        df: &DataFrame,
        row_idx: usize,
        column: &str,
        dtype: &DataType,
        seen_hashes: &mut HashSet<String>,
    ) -> Result<Vec<ModalitySegment>> {
        let mut segments = Vec::new();

        match dtype {
            DataType::Binary => {
                if let Some(bytes) = extract_binary_value(df, row_idx, column)? {
                    let hash = hash_bytes(&bytes);
                    if seen_hashes.insert(hash.clone()) {
                    segments.push(ModalitySegment {
                        modality_type: ModalityType::Image,
                        content: SegmentContent::Bytes(bytes),
                        metadata: Some(json!({
                            "column": column,
                            "source": "generic_processor",
                            "kind": "binary",
                                "hash": hash,
                        })),
                    });
                    }
                }
            }
            DataType::List(_) => {
                if let Some(JsonValue::Array(items)) = extract_list_json(df, row_idx, column)? {
                    for (idx, item) in items.into_iter().enumerate() {
                        match item {
                            JsonValue::String(path) => {
                                let content = if looks_like_url(&path) {
                                    SegmentContent::Url(path.clone())
                                } else {
                                    SegmentContent::FilePath(path.clone())
                                };

                                segments.push(ModalitySegment {
                                    modality_type: ModalityType::Image,
                                    content,
                                    metadata: Some(json!({
                                        "column": column,
                                        "index": idx,
                                        "source": "generic_processor",
                                    })),
                                });
                            }
                            _ => {
                                let bytes = serde_json::to_vec(&item)?;
                                let hash = hash_bytes(&bytes);
                                if seen_hashes.insert(hash.clone()) {
                                segments.push(ModalitySegment {
                                    modality_type: ModalityType::Image,
                                    content: SegmentContent::Bytes(bytes),
                                    metadata: Some(json!({
                                        "column": column,
                                        "index": idx,
                                        "source": "generic_processor",
                                            "hash": hash,
                                    })),
                                });
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                if let Some(text) = extract_text_value(df, row_idx, column)? {
                    if !text.is_empty() {
                        if let Ok(JsonValue::Array(values)) =
                            serde_json::from_str::<JsonValue>(&text)
                        {
                            for (idx, value) in values.into_iter().enumerate() {
                                if let Some(path) = value.as_str() {
                                    let content = if looks_like_url(path) {
                                        SegmentContent::Url(path.to_string())
                                    } else {
                                        SegmentContent::FilePath(path.to_string())
                                    };

                                    segments.push(ModalitySegment {
                                        modality_type: ModalityType::Image,
                                        content,
                                        metadata: Some(json!({
                                            "column": column,
                                            "index": idx,
                                            "source": "generic_processor",
                                            "kind": "reference",
                                        })),
                                    });
                                }
                            }
                        } else {
                    let content = if looks_like_url(&text) {
                        SegmentContent::Url(text.clone())
                    } else {
                        SegmentContent::FilePath(text.clone())
                    };

                    segments.push(ModalitySegment {
                        modality_type: ModalityType::Image,
                        content,
                        metadata: Some(json!({
                            "column": column,
                            "source": "generic_processor",
                            "kind": "reference",
                        })),
                    });
                        }
                    }
                }
            }
        }

        Ok(segments)
    }

    fn extract_textual_segments(
        &self,
        df: &DataFrame,
        row_idx: usize,
        column: &str,
        modality: &ModalityType,
        dtype: &DataType,
    ) -> Result<Vec<ModalitySegment>> {
        let mut segments = Vec::new();

        let payload = if matches!(dtype, DataType::List(_)) {
            extract_list_json(df, row_idx, column)?
                .map(|value| serde_json::to_vec(&value))
                .transpose()?
        } else if matches!(dtype, DataType::Struct(_)) {
            extract_struct_json(df, row_idx, column)?
                .map(|value| serde_json::to_vec(&value))
                .transpose()?
        } else {
            extract_text_value(df, row_idx, column)?.map(|text| text.into_bytes())
        };

        if let Some(bytes) = payload {
            if bytes.is_empty() {
                return Ok(segments);
            }

            let mut metadata = json!({
                "column": column,
                "source": "generic_processor",
            });

            if let Ok(text) = String::from_utf8(bytes.clone()) {
                if let Some(bbox_meta) = build_bbox_metadata(&text) {
                    if let Some(obj) = metadata.as_object_mut() {
                        obj.extend(bbox_meta);
                    }
                }
            }

            segments.push(ModalitySegment {
                modality_type: modality.clone(),
                content: SegmentContent::Bytes(bytes),
                metadata: Some(metadata),
            });
        }

        Ok(segments)
    }

    fn collect_reference_targets(&self, df: &DataFrame, row_idx: usize) -> Result<Vec<String>> {
        let mut targets = HashSet::new();

        for column_name in df.get_column_names() {
            if !looks_like_reference_column(column_name) {
                continue;
            }

            if let Some(JsonValue::Array(values)) = extract_list_json(df, row_idx, column_name)? {
                for value in values {
                    if let Some(candidate) = value_to_reference(&value) {
                        targets.insert(candidate);
                    }
                }
                continue;
            }

            if let Some(text) = extract_text_value(df, row_idx, column_name)? {
                if looks_like_url(&text) {
                    targets.insert(text);
                    continue;
                }

                if let Ok(JsonValue::Array(values)) = serde_json::from_str::<JsonValue>(&text) {
                    for value in values {
                        if let Some(candidate) = value_to_reference(&value) {
                            targets.insert(candidate);
                        }
                    }
                }
            }
        }

        Ok(targets.into_iter().collect())
    }

    fn download_url(&self, url: &str, dest: &Path) -> Result<()> {
        if dest.exists() {
            return Ok(());
        }

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        let response = self.http_client.get(url).send()?;
        let bytes = response.bytes()?;
        fs::write(dest, &bytes)?;
        Ok(())
    }

    fn filename_from_url(&self, url: &str) -> String {
        if let Some(filename) = url.split('/').last() {
            filename.to_string()
        } else {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(url.as_bytes());
            let hash = format!("{:x}", hasher.finalize());
            format!("file_{}", &hash[..16])
        }
    }

    fn download_hf_reference(&self, uri: &str, dataset_cache: &Path) -> Result<Option<PathBuf>> {
        let parsed = match parse_hf_uri_reference(uri) {
            Some(parsed) => parsed,
            None => return Ok(None),
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            parsed.repo_id.clone(),
            RepoType::Dataset,
            parsed.revision.clone(),
        ));

        let relative = sanitize_reference_path(&parsed.path);
        if relative.as_os_str().is_empty() {
            return Ok(None);
        }

        let dest = dataset_cache.join(&relative);
        if dest.exists() {
            return Ok(Some(dest));
        }

        let encoded_path = encode_path_segments(&parsed.path);
        let debug_url = repo.url(&encoded_path);
        println!("    Downloading hf reference {} via {}", uri, debug_url);

        match repo.get(&encoded_path) {
            Ok(source) => {
                copy_to_cache(&source, &dest)?;
                Ok(Some(dest))
            }
            Err(err) => {
                println!(
                    "    Warning: repo.get failed for {} ({:?}), falling back to HTTP",
                    uri, err
                );
                match self.download_via_repo_http(&repo, &encoded_path, &dest) {
                    Ok(_) => Ok(Some(dest)),
                    Err(http_err) => {
                        println!(
                            "    Warning: HTTP fallback failed for {}: {}",
                            uri, http_err
                        );
                        Ok(None)
                    }
                }
            }
        }
    }

    fn resolve_reference(
        &self,
        target: &str,
        dataset_cache: &Path,
        cache_root: &Path,
    ) -> Result<Option<DownloadedFile>> {
        if looks_like_url(target) {
            let filename = self.filename_from_url(target);
            let dest = dataset_cache.join(&filename);
            if let Err(err) = self.download_url(target, &dest) {
                eprintln!("Failed to download {}: {}", target, err);
                return Ok(None);
            }

            if let Ok(metadata) = fs::metadata(&dest) {
                return Ok(Some(DownloadedFile {
                    url: target.to_string(),
                    local_path: dest,
                    file_type: detect_file_type(target).to_string(),
                    size_bytes: Some(metadata.len()),
                }));
            }
            return Ok(None);
        }

        if target.starts_with("hf://") {
            match self.download_hf_reference(target, dataset_cache) {
                Ok(Some(local_path)) => {
                    let size = fs::metadata(&local_path).ok().map(|m| m.len());
                    return Ok(Some(DownloadedFile {
                        url: target.to_string(),
                        local_path,
                        file_type: detect_file_type(target).to_string(),
                        size_bytes: size,
                    }));
                }
                Ok(None) => {
                    println!("    Warning: Unable to resolve {}", target);
                    return Ok(None);
                }
                Err(err) => {
                    eprintln!("Failed to download {}: {}", target, err);
                    return Ok(None);
                }
            }
        }

        match self.download_dataset_asset(target, dataset_cache, cache_root) {
            Ok(Some(local_path)) => {
                let size = fs::metadata(&local_path).ok().map(|m| m.len());
                Ok(Some(DownloadedFile {
                    url: target.to_string(),
                    local_path,
                    file_type: detect_file_type(target).to_string(),
                    size_bytes: size,
                }))
            }
            Ok(None) => {
                println!("    Warning: Unable to resolve {}", target);
                Ok(None)
            }
            Err(err) => {
                eprintln!("Failed to download {}: {}", target, err);
                Ok(None)
            }
        }
    }

    fn download_dataset_asset(
        &self,
        reference: &str,
        dataset_cache: &Path,
        cache_root: &Path,
    ) -> Result<Option<PathBuf>> {
        let relative = sanitize_reference_path(reference);
        if relative.as_os_str().is_empty() {
            return Ok(None);
        }

        let dest = dataset_cache.join(&relative);
        if dest.exists() {
            return Ok(Some(dest));
        }

        if let Some(local_source) = find_local_dataset_asset(&self.dataset_name, &relative) {
            copy_to_cache(&local_source, &dest)?;
            return Ok(Some(dest));
        }

        let api = Api::new()?;
        let repo = api.repo(Repo::new(self.dataset_name.clone(), RepoType::Dataset));
        let rel_str = relative.to_string_lossy().to_string();
        let candidates = [&rel_str, &format!("data/{}", rel_str)];

        for candidate in candidates {
            let encoded_candidate = encode_path_segments(candidate);
            let debug_url = repo.url(&encoded_candidate);
            println!(
                "    Downloading dataset asset {} via {}",
                candidate, debug_url
            );
            match repo.get(&encoded_candidate) {
                Ok(source) => {
                    copy_to_cache(&source, &dest)?;
                    return Ok(Some(dest));
                }
                Err(err) => {
                    println!(
                        "    Warning: repo.get failed for {} ({:?}), falling back to HTTP",
                        candidate, err
                    );
                    if self
                        .download_via_repo_http(&repo, &encoded_candidate, &dest)
                        .is_ok()
                    {
                        return Ok(Some(dest));
                    }
                }
            }
        }

        if let Some(path) =
            self.try_external_sources(reference, &relative, dataset_cache, cache_root)?
        {
            return Ok(Some(path));
        }

        Ok(None)
    }

    fn download_via_repo_http(
        &self,
        repo: &ApiRepo,
        encoded_path: &str,
        dest: &Path,
    ) -> Result<()> {
        let url = repo.url(encoded_path);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }

        let response = self.http_client.get(url).send()?;
        if !response.status().is_success() {
            anyhow::bail!("HTTP {} fetching {}", response.status(), encoded_path);
        }

        let bytes = response.bytes()?;
        fs::write(dest, &bytes)?;
        Ok(())
    }

    fn try_external_sources(
        &self,
        reference: &str,
        sanitized_reference: &Path,
        dataset_cache: &Path,
        cache_root: &Path,
    ) -> Result<Option<PathBuf>> {
        let sources = matching_sources(&self.dataset_name, reference);
        if sources.is_empty() {
            return Ok(None);
        }

        for source in sources {
            if let Some(path) = self.download_from_external_source(
                source,
                reference,
                sanitized_reference,
                dataset_cache,
                cache_root,
            )? {
                return Ok(Some(path));
            }
        }

        Ok(None)
    }

    fn download_from_external_source(
        &self,
        source: &ExternalArchiveSource,
        reference: &str,
        sanitized_reference: &Path,
        dataset_cache: &Path,
        _cache_root: &Path,
    ) -> Result<Option<PathBuf>> {
        let trimmed = reference
            .strip_prefix(source.strip_prefix)
            .unwrap_or(reference)
            .trim_start_matches('/');
        if trimmed.is_empty() {
            return Ok(None);
        }

        let trimmed_path = sanitize_reference_path(trimmed);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(
            source.remote_dataset.to_string(),
            RepoType::Dataset,
        ));

        for archive in source.archives {
            if let Some(path) = self.fetch_from_external_archive(
                source.remote_dataset,
                &repo,
                archive,
                &trimmed_path,
                dataset_cache,
                sanitized_reference,
            )? {
                return Ok(Some(path));
            }
        }

        Ok(None)
    }

    fn fetch_from_external_archive(
        &self,
        remote_dataset: &str,
        repo: &ApiRepo,
        archive_path: &str,
        trimmed_path: &Path,
        dataset_cache: &Path,
        sanitized_reference: &Path,
    ) -> Result<Option<PathBuf>> {
        let encoded_archive = encode_path_segments(archive_path);
        let debug_url = repo.url(&encoded_archive);
        println!(
            "      Searching {} inside {} via {}",
            trimmed_path.display(),
            archive_path,
            debug_url
        );

        let source_path = match repo.get(&encoded_archive) {
            Ok(path) => path,
            Err(err) => {
                println!(
                    "      Warning: unable to fetch archive {} from {} ({:?})",
                    archive_path, remote_dataset, err
                );
                return Ok(None);
            }
        };

        let file = File::open(&source_path)?;
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        for entry_result in archive.entries()? {
            let mut entry = entry_result?;
            let entry_path = entry.path()?;
            if entry_path.as_ref() == trimmed_path {
                let dest = dataset_cache.join(sanitized_reference);
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent)?;
                }
                entry.unpack(&dest)?;
                return Ok(Some(dest));
            }
        }

        Ok(None)
    }
}

fn build_bbox_metadata(text: &str) -> Option<JsonMap<String, JsonValue>> {
    let bboxes = extract_bboxes_from_text(text);
    if bboxes.is_empty() {
        return None;
    }

    let mut map = JsonMap::new();
    map.insert("bbox_count".to_string(), JsonValue::from(bboxes.len()));
    map.insert("bounding_boxes".to_string(), serialize_bboxes(&bboxes));
    map.insert(
        "bbox_areas".to_string(),
        JsonValue::Array(
            bboxes
                .iter()
                .map(|bbox| JsonValue::from(bbox.area()))
                .collect(),
        ),
    );
    map.insert(
        "bbox_centroids".to_string(),
        JsonValue::Array(
            bboxes
                .iter()
                .map(|bbox| {
                    let (cx, cy) = bbox.centroid();
                    json!({ "cx": cx, "cy": cy })
                })
                .collect(),
        ),
    );

    Some(map)
}

fn value_to_reference(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(s) => Some(s.to_string()),
        JsonValue::Object(obj) => {
            if let Some(path) = obj
                .get("url")
                .or_else(|| obj.get("path"))
                .and_then(|v| v.as_str())
            {
                Some(path.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn sanitize_reference_path(reference: &str) -> PathBuf {
    let mut path = PathBuf::new();
    for segment in reference.split(['/', '\\']) {
        if segment.is_empty() || segment == "." || segment == ".." {
            continue;
        }
        path.push(segment);
    }
    path
}

fn encode_path_segments(path: &str) -> String {
    path.split('/')
        .map(|segment| encode(segment).to_string())
        .collect::<Vec<_>>()
        .join("/")
}

fn copy_to_cache(source: &Path, dest: &Path) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::copy(source, dest)?;
    Ok(())
}

fn find_local_dataset_asset(dataset_name: &str, relative: &Path) -> Option<PathBuf> {
    dataset_cache_dir(dataset_name)
        .ok()
        .map(|dir| dir.join(relative))
        .filter(|candidate| candidate.exists())
}

#[derive(Clone)]
struct ParsedHfUri {
    repo_id: String,
    revision: String,
    path: String,
}

fn parse_hf_uri_reference(uri: &str) -> Option<ParsedHfUri> {
    const PREFIX: &str = "hf://datasets/";
    let rest = uri.strip_prefix(PREFIX)?;
    let mut parts = rest.splitn(3, '/');
    let owner = parts.next()?;
    let dataset_part = parts.next()?;
    let path = parts.next().unwrap_or("").to_string();
    if path.is_empty() {
        return None;
    }

    let (dataset_name, revision) = if let Some(idx) = dataset_part.find('@') {
        (
            dataset_part[..idx].to_string(),
            dataset_part[idx + 1..].to_string(),
        )
    } else {
        (dataset_part.to_string(), "main".to_string())
    };

    Some(ParsedHfUri {
        repo_id: format!("{}/{}", owner, dataset_name),
        revision,
        path,
    })
}

fn infer_modality(column_name: &str, dtype: &DataType) -> ModalityType {
    let name = column_name.to_lowercase();

    if name.contains("image") || matches!(dtype, DataType::Binary) {
        ModalityType::Image
    } else if name.contains("audio") {
        ModalityType::Audio
    } else if name.contains("video") {
        ModalityType::Video
    } else if name.contains("pdf") || name.contains("doc") || name.contains("html") {
        ModalityType::Document
    } else if name.contains("text")
        || name.contains("caption")
        || name.contains("conversation")
        || matches!(dtype, DataType::List(_))
        || matches!(dtype, DataType::Struct(_))
    {
        ModalityType::Text
    } else {
        ModalityType::Unknown(column_name.to_string())
    }
}

fn looks_like_url(value: &str) -> bool {
    value.starts_with("http://")
        || value.starts_with("https://")
        || value.starts_with("s3://")
        || value.starts_with("gs://")
}

fn looks_like_reference_column(column: &str) -> bool {
    let name = column.to_lowercase();
    name.contains("url")
        || name.contains("reference")
        || name.contains("file")
        || name.contains("path")
        || name.contains("image")
        || name.contains("media")
        || name.contains("asset")
}

fn detect_file_type(path: &str) -> &'static str {
    let lower = path.to_lowercase();
    for (suffix, kind) in [
        (".wav", "wav"),
        (".mp3", "mp3"),
        (".flac", "flac"),
        (".jpg", "jpg"),
        (".jpeg", "jpg"),
        (".png", "png"),
        (".gif", "gif"),
        (".mp4", "mp4"),
        (".avi", "avi"),
        (".mov", "mov"),
        (".pdf", "pdf"),
        (".docx", "docx"),
        (".xlsx", "xlsx"),
        (".json", "json"),
    ] {
        if lower.ends_with(suffix) {
            return kind;
        }
    }
    "unknown"
}

impl DatasetItemProcessor for GenericDatasetProcessor {
    fn process_item(&self, df: &DataFrame, row_idx: usize) -> Result<Vec<ModalitySegment>> {
        self.process_row(df, row_idx)
    }

    fn dataset_metadata(&self) -> DatasetMetadata {
        DatasetMetadata {
            name: self.dataset_name.clone(),
            version: "generic-1.0".to_string(),
            description: "Schema-inferred Polars dataset processor".to_string(),
            requires_download: true,
            modalities: vec![
                ModalityType::Text,
                ModalityType::Image,
                ModalityType::Audio,
                ModalityType::Video,
                ModalityType::Document,
            ],
        }
    }

    fn download_references(
        &self,
        df: &DataFrame,
        row_idx: usize,
        cache_dir: &Path,
    ) -> Result<Vec<DownloadedFile>> {
        let targets = self.collect_reference_targets(df, row_idx)?;
        if targets.is_empty() {
            return Ok(Vec::new());
        }

        let dataset_slug = self.dataset_name.replace('/', "_");
        let dataset_cache = cache_dir.join(&dataset_slug);
        fs::create_dir_all(&dataset_cache)?;

        let mut downloaded = Vec::new();

        for target in targets {
            if let Some(file) = self.resolve_reference(&target, &dataset_cache, cache_dir)? {
                downloaded.push(file);
            }
        }

        Ok(downloaded)
    }

    fn prefetch_references(&self, df: &DataFrame, cache_dir: &Path) -> Result<()> {
        let mut all_targets = HashSet::new();
        for row_idx in 0..df.height() {
            let references = self.collect_reference_targets(df, row_idx)?;
            all_targets.extend(references);
        }

        if all_targets.is_empty() {
            return Ok(());
        }

        let dataset_slug = self.dataset_name.replace('/', "_");
        let dataset_cache = cache_dir.join(&dataset_slug);
        fs::create_dir_all(&dataset_cache)?;

        println!(
            "Prefetching {} unique references for {}",
            all_targets.len(),
            self.dataset_name
        );

        for target in all_targets {
            let _ = self.resolve_reference(&target, &dataset_cache, cache_dir)?;
        }

        Ok(())
    }
}
