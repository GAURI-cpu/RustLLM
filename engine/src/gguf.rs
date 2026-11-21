//! My own GGUF reader: opens a .gguf model file without any ML library.
//!
//! File layout:
//!   HEADER       "GGUF", version (u32), tensor count (u64), metadata count (u64)
//!   METADATA     key (string) + value type (u32) + value, repeated
//!   TENSOR INFO  name, dims, type, offset, repeated
//!   padding      up to the alignment (32 bytes unless the file says otherwise)
//!   TENSOR DATA  the weights

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use anyhow::Context;

/// One metadata value. GGUF has 13 kinds.
#[derive(Debug)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// Where one tensor (one weight matrix) is, and what it looks like.
#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>, // shape, e.g. [1536, 1536]
    pub kind: u32,      // 0 = F32, 1 = F16, 2 = Q4_0, 8 = Q8_0, 12 = Q4_K, 14 = Q6_K ...
    pub offset: u64,    // where its bytes start, counted from data_start
}

/// Everything the file says about itself (not the weights yet).
pub struct Gguf {
    pub path: String,
    pub version: u32,
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data_start: u64, // byte position where the tensor data begins
}

// ---------- Step 1.1: numbers ----------

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4]; // a box for 4 bytes
    r.read_exact(&mut buf)?; // fill it from the file
    Ok(u32::from_le_bytes(buf)) // 4 bytes → one number, little-endian
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8]; // u64 = 8 bytes
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

// ---------- Step 1.2: text ----------

fn read_string(r: &mut impl Read) -> std::io::Result<String> {
    let len = read_u64(r)?; // how many bytes the text has
    let mut bytes = vec![0u8; len as usize]; // a box of exactly that size
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ---------- Step 1.3: any metadata value ----------

/// Read exactly N bytes (N is worked out from how the bytes are used).
fn read_bytes<const N: usize>(r: &mut impl Read) -> std::io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Read one metadata value. `kind` (from the file) says which of the 13 kinds it is.
fn read_value(r: &mut impl Read, kind: u32) -> std::io::Result<Value> {
    let value = match kind {
        0 => Value::U8(u8::from_le_bytes(read_bytes(r)?)),
        1 => Value::I8(i8::from_le_bytes(read_bytes(r)?)),
        2 => Value::U16(u16::from_le_bytes(read_bytes(r)?)),
        3 => Value::I16(i16::from_le_bytes(read_bytes(r)?)),
        4 => Value::U32(read_u32(r)?),
        5 => Value::I32(i32::from_le_bytes(read_bytes(r)?)),
        6 => Value::F32(f32::from_le_bytes(read_bytes(r)?)),
        7 => Value::Bool(u8::from_le_bytes(read_bytes(r)?) != 0),
        8 => Value::Str(read_string(r)?),
        9 => {
            // A list: what kind its items are, how many, then the items
            let item_kind = read_u32(r)?;
            let count = read_u64(r)?;
            let mut items = Vec::new();
            for _ in 0..count {
                items.push(read_value(r, item_kind)?); // the function calls itself for each item
            }
            Value::Array(items)
        }
        10 => Value::U64(read_u64(r)?),
        11 => Value::I64(i64::from_le_bytes(read_bytes(r)?)),
        12 => Value::F64(f64::from_le_bytes(read_bytes(r)?)),
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown metadata value type {other}"),
            ));
        }
    };
    Ok(value)
}

// ---------- Step 1.4: the whole file ----------

impl Gguf {
    pub fn open(path: &str) -> anyhow::Result<Gguf> {
        let file = std::fs::File::open(path).with_context(|| format!("could not open {path}"))?;
        let mut r = std::io::BufReader::new(file); // reads in big chunks: much faster

        // 1. Cover: magic word, version, how many tensors and info entries
        let magic: [u8; 4] = read_bytes(&mut r)?;
        if &magic != b"GGUF" {
            anyhow::bail!("{path} is not a GGUF file");
        }
        let version = read_u32(&mut r)?;
        if version < 2 {
            anyhow::bail!("GGUF version {version} is too old (this reader supports 2 and 3)");
        }
        let tensor_count = read_u64(&mut r)?;
        let metadata_count = read_u64(&mut r)?;

        // 2. Info page: key -> value
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = read_string(&mut r)?;
            let kind = read_u32(&mut r)?;
            let value = read_value(&mut r, kind).with_context(|| format!("reading metadata '{key}'"))?;
            metadata.insert(key, value);
        }

        // 3. Index: one entry per tensor
        let mut tensors = Vec::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)?;
            let mut dims = Vec::new();
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let kind = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensors.push(TensorInfo { name, dims, kind, offset });
        }

        // 4. Padding: the data starts at the next multiple of the alignment
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U32(a)) => *a as u64,
            _ => 32,
        };
        let position = r.stream_position()?; // how many bytes we've read so far
        let data_start = position.div_ceil(alignment) * alignment;

        Ok(Gguf { path: path.to_string(), version, metadata, tensors, data_start })
    }

    // ---------- Step 2.4: one tensor's weights ----------

    /// One tensor's weights as normal f32 numbers.
    pub fn read_tensor_f32(&self, name: &str) -> anyhow::Result<Vec<f32>> {
        // 1. Find it in the index
        let info = self.tensors.iter().find(|t| t.name == name).with_context(|| format!("no tensor called {name}"))?;
        let count: u64 = info.dims.iter().product();

        // 2. How many bytes it takes in the file
        let (per_block, block_bytes) = crate::quant::block_info(info.kind).with_context(|| {
            format!("{name}: storage type {} isn't supported yet", crate::quant::kind_name(info.kind))
        })?;
        let size = count / per_block as u64 * block_bytes as u64;

        // 3. Jump to its bytes and read them
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.data_start + info.offset))?;
        let mut bytes = vec![0u8; size as usize];
        file.read_exact(&mut bytes)?;

        // 4. Unpack
        let mut out = Vec::with_capacity(count as usize);
        crate::quant::dequantize(info.kind, &bytes, &mut out);
        Ok(out)
    }

    /// A weight matrix, kept packed (4-bit stays 4-bit in memory).
    pub fn matrix(&self, name: &str) -> anyhow::Result<crate::quant::QMatrix> {
        let info = self.tensors.iter().find(|t| t.name == name).with_context(|| format!("no tensor called {name}"))?;
        let cols = info.dims[0] as usize; // GGUF lists the shape backwards: numbers per row first
        let rows: usize = info.dims[1..].iter().product::<u64>() as usize;
        let row_bytes = crate::quant::row_bytes(info.kind, cols)
            .with_context(|| format!("{name}: storage type {} isn't supported yet", crate::quant::kind_name(info.kind)))?;
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.data_start + info.offset))?;
        let mut bytes = vec![0u8; rows * row_bytes];
        file.read_exact(&mut bytes)?;
        Ok(crate::quant::QMatrix { kind: info.kind, rows, cols, bytes })
    }

    /// A decimal setting, e.g. get_f32("qwen2.rope.freq_base") -> Some(1000000.0)
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            Value::F32(v) => Some(*v),
            _ => None,
        }
    }
    

    /// A whole-number setting, e.g. get_u32("qwen2.block_count") -> Some(28)
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            Value::U32(v) => Some(*v),
            _ => None,
        }
    }

    /// A text setting, e.g. get_str("general.architecture") -> Some("qwen2")
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}