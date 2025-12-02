use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct BBox {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

impl BBox {
    pub fn area(&self) -> f64 {
        let width = (self.x2 - self.x1).abs();
        let height = (self.y2 - self.y1).abs();
        width * height
    }

    pub fn centroid(&self) -> (f64, f64) {
        let cx = f64::midpoint(self.x1, self.x2);
        let cy = f64::midpoint(self.y1, self.y2);
        (cx, cy)
    }

    pub fn normalize(&self, width: f64, height: f64) -> Self {
        Self {
            x1: (self.x1 / width).clamp(0.0, 1.0),
            y1: (self.y1 / height).clamp(0.0, 1.0),
            x2: (self.x2 / width).clamp(0.0, 1.0),
            y2: (self.y2 / height).clamp(0.0, 1.0),
        }
    }
}

static BOX_RE: OnceLock<Regex> = OnceLock::new();

/// Extract bounding boxes encoded inside text spans, e.g. `<box>[x1,y1,x2,y2]</box>`.
///
/// # Panics
/// Panics if the internal bounding box regex is invalid (compile-time constant, should never fail).
pub fn extract_bboxes_from_text(text: &str) -> Vec<BBox> {
    let regex = BOX_RE.get_or_init(|| {
        Regex::new(
            r"<box>\s*\[(?P<x1>-?\d+(?:\.\d+)?),(?P<y1>-?\d+(?:\.\d+)?),(?P<x2>-?\d+(?:\.\d+)?),(?P<y2>-?\d+(?:\.\d+)?)\]\s*</box>",
        )
        .expect("valid bounding box regex")
    });

    regex
        .captures_iter(text)
        .filter_map(|cap| {
            Some(BBox {
                x1: cap.name("x1")?.as_str().parse().ok()?,
                y1: cap.name("y1")?.as_str().parse().ok()?,
                x2: cap.name("x2")?.as_str().parse().ok()?,
                y2: cap.name("y2")?.as_str().parse().ok()?,
            })
        })
        .collect()
}

/// Serialize bounding boxes into JSON for metadata emission.
pub fn serialize_bboxes(bboxes: &[BBox]) -> Value {
    Value::Array(
        bboxes
            .iter()
            .map(|bbox| {
                Value::Object(serde_json::Map::from_iter([
                    ("x1".to_string(), Value::from(bbox.x1)),
                    ("y1".to_string(), Value::from(bbox.y1)),
                    ("x2".to_string(), Value::from(bbox.x2)),
                    ("y2".to_string(), Value::from(bbox.y2)),
                ]))
            })
            .collect(),
    )
}
