#if 0
;
; Note: shader requires additional functionality:
;       Use native low precision
;
;
; Input signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
;
; Output signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
; shader hash: 7b12544393df7288a6f41448572b93d5
;
; Pipeline Runtime Information: 
;
;
;
; Buffer Definitions:
;
; cbuffer 
; {
;
;   [88 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [2 x i8] (type annotation not present)
;
; }
;
; Resource bind info for 
; {
;
;   [2 x i8] (type annotation not present)
;
; }
;
;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ------
;                                   cbuffer      NA          NA     CB0            cb0     1
;                                       UAV  struct         r/w      U0             u0     1
;                                       UAV  struct         r/w      U1             u1     1
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

%dx.types.Handle = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.ResRet.f16 = type { half, half, half, half, i32 }
%dx.types.CBufRet.f32 = type { float, float, float, float }
%"class.RWStructuredBuffer<half>" = type { half }
%Constants = type { i32, i32, i32, i32, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, float, i32 }

define void @DFT() {
  %1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 1, i32 1, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %3 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %4 = call i32 @dx.op.threadId.i32(i32 93, i32 0)  ; ThreadId(component)
  %5 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %6 = extractvalue %dx.types.CBufRet.i32 %5, 0
  %7 = add i32 %6, %4
  %8 = extractvalue %dx.types.CBufRet.i32 %5, 1
  %9 = icmp ult i32 %7, %8
  br i1 %9, label %10, label %103

; <label>:10                                      ; preds = %0
  %11 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 5)  ; CBufferLoadLegacy(handle,regIndex)
  %12 = extractvalue %dx.types.CBufRet.i32 %11, 1
  %13 = lshr i32 %12, 1
  %14 = extractvalue %dx.types.CBufRet.i32 %5, 2
  %15 = and i32 %14, 31
  %16 = shl i32 1, %15
  %17 = add i32 %14, 31
  %18 = and i32 %17, 31
  %19 = shl nuw i32 1, %18
  %20 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 3)  ; CBufferLoadLegacy(handle,regIndex)
  %21 = extractvalue %dx.types.CBufRet.i32 %20, 1
  %22 = extractvalue %dx.types.CBufRet.i32 %20, 2
  %23 = mul i32 %22, %21
  %24 = urem i32 %7, %23
  %25 = udiv i32 %7, %23
  %26 = udiv i32 %24, %22
  %27 = urem i32 %24, %22
  %28 = lshr i32 %26, %15
  %29 = shl i32 %28, %18
  %30 = add i32 %19, -1
  %31 = and i32 %26, %30
  %32 = add i32 %29, %31
  %33 = add i32 %32, %13
  %34 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %35 = extractvalue %dx.types.CBufRet.i32 %34, 0
  %36 = mul i32 %35, %25
  %37 = extractvalue %dx.types.CBufRet.i32 %34, 1
  %38 = mul i32 %32, %37
  %39 = extractvalue %dx.types.CBufRet.i32 %34, 2
  %40 = mul i32 %39, %27
  %41 = add i32 %40, %36
  %42 = add i32 %41, %38
  %43 = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %2, i32 %42, i32 0, i8 1, i32 2)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %44 = extractvalue %dx.types.ResRet.f16 %43, 0
  %45 = fpext half %44 to float
  %46 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %47 = extractvalue %dx.types.CBufRet.i32 %46, 3
  %48 = icmp eq i32 %47, 2
  br i1 %48, label %49, label %55

; <label>:49                                      ; preds = %10
  %50 = extractvalue %dx.types.CBufRet.i32 %34, 3
  %51 = add i32 %50, %42
  %52 = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %2, i32 %51, i32 0, i8 1, i32 2)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %53 = extractvalue %dx.types.ResRet.f16 %52, 0
  %54 = fpext half %53 to float
  br label %55

; <label>:55                                      ; preds = %49, %10
  %56 = phi float [ %54, %49 ], [ 0.000000e+00, %10 ]
  %57 = mul i32 %37, %33
  %58 = add i32 %57, %36
  %59 = add i32 %58, %40
  %60 = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %2, i32 %59, i32 0, i8 1, i32 2)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %61 = extractvalue %dx.types.ResRet.f16 %60, 0
  %62 = fpext half %61 to float
  br i1 %48, label %63, label %69

; <label>:63                                      ; preds = %55
  %64 = extractvalue %dx.types.CBufRet.i32 %34, 3
  %65 = add i32 %64, %59
  %66 = call %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32 139, %dx.types.Handle %2, i32 %65, i32 0, i8 1, i32 2)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %67 = extractvalue %dx.types.ResRet.f16 %66, 0
  %68 = fpext half %67 to float
  br label %69

; <label>:69                                      ; preds = %63, %55
  %70 = phi float [ %68, %63 ], [ 0.000000e+00, %55 ]
  %71 = add i32 %16, -1
  %72 = and i32 %26, %71
  %73 = extractvalue %dx.types.CBufRet.i32 %5, 3
  %74 = icmp eq i32 %73, 1
  %75 = select i1 %74, float 0x401921FB60000000, float 0xC01921FB60000000
  %76 = uitofp i32 %72 to float
  %77 = fmul fast float %75, %76
  %78 = uitofp i32 %16 to float
  %79 = fdiv fast float %77, %78
  %80 = call float @dx.op.unary.f32(i32 12, float %79)  ; Cos(value)
  %81 = call float @dx.op.unary.f32(i32 13, float %79)  ; Sin(value)
  %82 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 4)  ; CBufferLoadLegacy(handle,regIndex)
  %83 = extractvalue %dx.types.CBufRet.i32 %82, 2
  %84 = mul i32 %83, %7
  %85 = extractvalue %dx.types.CBufRet.i32 %82, 3
  %86 = add i32 %84, %85
  %87 = call %dx.types.CBufRet.f32 @dx.op.cbufferLoadLegacy.f32(i32 59, %dx.types.Handle %3, i32 5)  ; CBufferLoadLegacy(handle,regIndex)
  %88 = extractvalue %dx.types.CBufRet.f32 %87, 0
  %89 = fmul fast float %80, %62
  %90 = fmul fast float %81, %70
  %91 = fadd fast float %89, %45
  %92 = fsub fast float %91, %90
  %93 = fmul fast float %88, %92
  %94 = fptrunc float %93 to half
  call void @dx.op.rawBufferStore.f16(i32 140, %dx.types.Handle %1, i32 %84, i32 0, half %94, half undef, half undef, half undef, i8 1, i32 2)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %95 = call %dx.types.CBufRet.f32 @dx.op.cbufferLoadLegacy.f32(i32 59, %dx.types.Handle %3, i32 5)  ; CBufferLoadLegacy(handle,regIndex)
  %96 = extractvalue %dx.types.CBufRet.f32 %95, 0
  %97 = fmul fast float %80, %70
  %98 = fmul fast float %81, %62
  %99 = fadd fast float %97, %56
  %100 = fadd fast float %99, %98
  %101 = fmul fast float %96, %100
  %102 = fptrunc float %101 to half
  call void @dx.op.rawBufferStore.f16(i32 140, %dx.types.Handle %1, i32 %86, i32 0, half %102, half undef, half undef, half undef, i8 1, i32 2)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %103

; <label>:103                                     ; preds = %69, %0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #0

; Function Attrs: nounwind readnone
declare float @dx.op.unary.f32(i32, float) #0

; Function Attrs: nounwind readonly
declare %dx.types.ResRet.f16 @dx.op.rawBufferLoad.f16(i32, %dx.types.Handle, i32, i32, i8, i32) #1

; Function Attrs: nounwind
declare void @dx.op.rawBufferStore.f16(i32, %dx.types.Handle, i32, i32, half, half, half, half, i8, i32) #2

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.f32 @dx.op.cbufferLoadLegacy.f32(i32, %dx.types.Handle, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind }

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.resources = !{!4}
!dx.entryPoints = !{!10}

!0 = !{!"clang version 3.7 (tags/RELEASE_370/final)"}
!1 = !{i32 1, i32 2}
!2 = !{i32 1, i32 6}
!3 = !{!"cs", i32 6, i32 2}
!4 = !{null, !5, !8, null}
!5 = !{!6, !7}
!6 = !{i32 0, %"class.RWStructuredBuffer<half>"* undef, !"", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !1}
!7 = !{i32 1, %"class.RWStructuredBuffer<half>"* undef, !"", i32 0, i32 1, i32 1, i32 12, i1 false, i1 false, i1 false, !1}
!8 = !{!9}
!9 = !{i32 0, %Constants* undef, !"", i32 0, i32 0, i32 1, i32 88, null}
!10 = !{void ()* @DFT, !"DFT", null, !4, !11}
!11 = !{i32 0, i64 8388656, i32 4, !12}
!12 = !{i32 64, i32 1, i32 1}

#endif

const unsigned char g_DFT[] = {
  0x44, 0x58, 0x42, 0x43, 0x98, 0xa9, 0x34, 0xb4, 0x5a, 0x27, 0x90, 0x50,
  0x87, 0xa4, 0xc0, 0xc4, 0xcb, 0x65, 0x3d, 0xa3, 0x01, 0x00, 0x00, 0x00,
  0x78, 0x09, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00,
  0x48, 0x00, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x68, 0x00, 0x00, 0x00,
  0x00, 0x01, 0x00, 0x00, 0x1c, 0x01, 0x00, 0x00, 0x53, 0x46, 0x49, 0x30,
  0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x49, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x4f, 0x53, 0x47, 0x31, 0x08, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x50, 0x53, 0x56, 0x30,
  0x90, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x05, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  0x18, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x48, 0x41, 0x53, 0x48, 0x14, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x7b, 0x12, 0x54, 0x43, 0x93, 0xdf, 0x72, 0x88,
  0xa6, 0xf4, 0x14, 0x48, 0x57, 0x2b, 0x93, 0xd5, 0x44, 0x58, 0x49, 0x4c,
  0x54, 0x08, 0x00, 0x00, 0x62, 0x00, 0x05, 0x00, 0x15, 0x02, 0x00, 0x00,
  0x44, 0x58, 0x49, 0x4c, 0x02, 0x01, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
  0x3c, 0x08, 0x00, 0x00, 0x42, 0x43, 0xc0, 0xde, 0x21, 0x0c, 0x00, 0x00,
  0x0c, 0x02, 0x00, 0x00, 0x0b, 0x82, 0x20, 0x00, 0x02, 0x00, 0x00, 0x00,
  0x13, 0x00, 0x00, 0x00, 0x07, 0x81, 0x23, 0x91, 0x41, 0xc8, 0x04, 0x49,
  0x06, 0x10, 0x32, 0x39, 0x92, 0x01, 0x84, 0x0c, 0x25, 0x05, 0x08, 0x19,
  0x1e, 0x04, 0x8b, 0x62, 0x80, 0x18, 0x45, 0x02, 0x42, 0x92, 0x0b, 0x42,
  0xc4, 0x10, 0x32, 0x14, 0x38, 0x08, 0x18, 0x4b, 0x0a, 0x32, 0x62, 0x88,
  0x48, 0x90, 0x14, 0x20, 0x43, 0x46, 0x88, 0xa5, 0x00, 0x19, 0x32, 0x42,
  0xe4, 0x48, 0x0e, 0x90, 0x11, 0x23, 0xc4, 0x50, 0x41, 0x51, 0x81, 0x8c,
  0xe1, 0x83, 0xe5, 0x8a, 0x04, 0x31, 0x46, 0x06, 0x51, 0x18, 0x00, 0x00,
  0x08, 0x00, 0x00, 0x00, 0x1b, 0x8c, 0xe0, 0xff, 0xff, 0xff, 0xff, 0x07,
  0x40, 0x02, 0xa8, 0x0d, 0x86, 0xf0, 0xff, 0xff, 0xff, 0xff, 0x03, 0x20,
  0x01, 0xd5, 0x06, 0x62, 0xf8, 0xff, 0xff, 0xff, 0xff, 0x01, 0x90, 0x00,
  0x49, 0x18, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x13, 0x82, 0x60, 0x42,
  0x20, 0x4c, 0x08, 0x06, 0x00, 0x00, 0x00, 0x00, 0x89, 0x20, 0x00, 0x00,
  0x40, 0x00, 0x00, 0x00, 0x32, 0x22, 0x88, 0x09, 0x20, 0x64, 0x85, 0x04,
  0x13, 0x23, 0xa4, 0x84, 0x04, 0x13, 0x23, 0xe3, 0x84, 0xa1, 0x90, 0x14,
  0x12, 0x4c, 0x8c, 0x8c, 0x0b, 0x84, 0xc4, 0x4c, 0x10, 0x8c, 0xc1, 0x08,
  0x40, 0x09, 0x00, 0x0a, 0xe6, 0x08, 0xc0, 0xa0, 0x0c, 0xc3, 0x30, 0x10,
  0x31, 0x03, 0x50, 0x06, 0x63, 0x30, 0xe8, 0x18, 0x05, 0xb8, 0x69, 0xb8,
  0xfc, 0x09, 0x7b, 0x08, 0xc9, 0x5f, 0x09, 0x69, 0x25, 0x26, 0xbf, 0xa8,
  0x75, 0x54, 0x24, 0x49, 0x92, 0x0c, 0x73, 0x04, 0x08, 0x2d, 0xf7, 0x0c,
  0x97, 0x3f, 0x61, 0x0f, 0x21, 0xf9, 0x21, 0xd0, 0x0c, 0x0b, 0x81, 0x02,
  0xa6, 0x1c, 0xca, 0xd0, 0x0c, 0xc3, 0x32, 0x90, 0x53, 0x16, 0x60, 0x68,
  0x86, 0x21, 0x49, 0x92, 0x64, 0x19, 0x08, 0x3a, 0x6a, 0xb8, 0xfc, 0x09,
  0x7b, 0x08, 0xc9, 0xe7, 0x36, 0xaa, 0x58, 0x89, 0xc9, 0x47, 0x6e, 0x1b,
  0x11, 0xc3, 0x30, 0x0c, 0x85, 0x90, 0x86, 0x66, 0xa0, 0xe9, 0xa8, 0xe1,
  0xf2, 0x27, 0xec, 0x21, 0x24, 0x9f, 0xdb, 0xa8, 0x62, 0x25, 0x26, 0xbf,
  0xb8, 0x6d, 0x44, 0x18, 0x86, 0x61, 0x14, 0xa2, 0x1a, 0x9a, 0x81, 0xac,
  0x39, 0x82, 0xa0, 0x18, 0xcd, 0xb0, 0x0c, 0x03, 0x46, 0xd9, 0x40, 0xc0,
  0x4c, 0xde, 0x38, 0xb0, 0x43, 0x38, 0xcc, 0xc3, 0x3c, 0xb8, 0x81, 0x2c,
  0xdc, 0xc2, 0x2c, 0xd0, 0x83, 0x3c, 0xd4, 0xc3, 0x38, 0xd0, 0x43, 0x3d,
  0xc8, 0x43, 0x39, 0x90, 0x83, 0x28, 0xd4, 0x83, 0x39, 0x98, 0x43, 0x39,
  0xc8, 0x03, 0x1f, 0xa0, 0x43, 0x38, 0xb0, 0x83, 0x39, 0xf8, 0x01, 0x0a,
  0x12, 0xe2, 0x86, 0x11, 0x88, 0xe1, 0x12, 0xce, 0x69, 0xa4, 0x09, 0x68,
  0x26, 0x09, 0x29, 0xc3, 0x30, 0x0c, 0x9e, 0xe7, 0x79, 0xc6, 0x40, 0xdf,
  0x1c, 0x01, 0x28, 0x4c, 0x01, 0x00, 0x00, 0x00, 0x13, 0x14, 0x72, 0xc0,
  0x87, 0x74, 0x60, 0x87, 0x36, 0x68, 0x87, 0x79, 0x68, 0x03, 0x72, 0xc0,
  0x87, 0x0d, 0xae, 0x50, 0x0e, 0x6d, 0xd0, 0x0e, 0x7a, 0x50, 0x0e, 0x6d,
  0x00, 0x0f, 0x7a, 0x30, 0x07, 0x72, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d,
  0x90, 0x0e, 0x71, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x90, 0x0e, 0x78,
  0xa0, 0x07, 0x78, 0xd0, 0x06, 0xe9, 0x10, 0x07, 0x76, 0xa0, 0x07, 0x71,
  0x60, 0x07, 0x6d, 0x90, 0x0e, 0x73, 0x20, 0x07, 0x7a, 0x30, 0x07, 0x72,
  0xd0, 0x06, 0xe9, 0x60, 0x07, 0x74, 0xa0, 0x07, 0x76, 0x40, 0x07, 0x6d,
  0x60, 0x0e, 0x71, 0x60, 0x07, 0x7a, 0x10, 0x07, 0x76, 0xd0, 0x06, 0xe6,
  0x30, 0x07, 0x72, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x6d, 0x60, 0x0e, 0x76,
  0x40, 0x07, 0x7a, 0x60, 0x07, 0x74, 0xd0, 0x06, 0xee, 0x80, 0x07, 0x7a,
  0x10, 0x07, 0x76, 0xa0, 0x07, 0x73, 0x20, 0x07, 0x7a, 0x60, 0x07, 0x74,
  0x30, 0xe4, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x60, 0xc8, 0x43, 0x00, 0x01, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0xc0, 0x90, 0xe7, 0x00, 0x02, 0x20, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x80, 0x21, 0x8f, 0x03, 0x04, 0x80, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x43, 0x1e, 0x08, 0x08, 0x80, 0x01, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x86, 0x3c, 0x13, 0x10, 0x00, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x79, 0x2c, 0x20, 0x00,
  0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0xf2, 0x64, 0x40,
  0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x05, 0x02,
  0x0c, 0x00, 0x00, 0x00, 0x32, 0x1e, 0x98, 0x14, 0x19, 0x11, 0x4c, 0x90,
  0x8c, 0x09, 0x26, 0x47, 0xc6, 0x04, 0x43, 0x1a, 0x4a, 0xa0, 0x08, 0x8a,
  0x61, 0x04, 0xa0, 0x30, 0x0a, 0x36, 0xa0, 0x10, 0x0a, 0x30, 0x80, 0xb0,
  0x11, 0x00, 0x0a, 0x0b, 0x1c, 0x10, 0x10, 0x81, 0xc0, 0x19, 0x00, 0xea,
  0x66, 0x00, 0x00, 0x00, 0x79, 0x18, 0x00, 0x00, 0x43, 0x00, 0x00, 0x00,
  0x1a, 0x03, 0x4c, 0x90, 0x46, 0x02, 0x13, 0x44, 0x35, 0x18, 0x63, 0x0b,
  0x73, 0x3b, 0x03, 0xb1, 0x2b, 0x93, 0x9b, 0x4b, 0x7b, 0x73, 0x03, 0x99,
  0x71, 0xb9, 0x01, 0x41, 0xa1, 0x0b, 0x3b, 0x9b, 0x7b, 0x91, 0x2a, 0x62,
  0x2a, 0x0a, 0x9a, 0x2a, 0xfa, 0x9a, 0xb9, 0x81, 0x79, 0x31, 0x4b, 0x73,
  0x0b, 0x63, 0x4b, 0xd9, 0x10, 0x04, 0x13, 0x84, 0x01, 0x99, 0x20, 0x0c,
  0xc9, 0x06, 0x61, 0x20, 0x26, 0x08, 0x83, 0xb2, 0x41, 0x18, 0x0c, 0x0a,
  0x63, 0x73, 0x1b, 0x06, 0xc4, 0x20, 0x26, 0x08, 0xc3, 0x32, 0x41, 0xe8,
  0x26, 0x02, 0x13, 0x84, 0x81, 0x99, 0x20, 0x60, 0xd0, 0x86, 0x45, 0x59,
  0x18, 0x45, 0x19, 0x1a, 0xc7, 0x71, 0x8a, 0x0d, 0xcb, 0xb0, 0x30, 0xca,
  0x30, 0x34, 0x8e, 0xe3, 0x14, 0x1b, 0x84, 0x07, 0x9a, 0x20, 0x80, 0x81,
  0x34, 0x41, 0x18, 0x9a, 0x0d, 0x88, 0x22, 0x31, 0x8a, 0x32, 0x4c, 0xc0,
  0x86, 0x80, 0xda, 0x40, 0x00, 0x51, 0x05, 0x4c, 0x10, 0x04, 0x80, 0x03,
  0x91, 0x11, 0xd5, 0x04, 0x21, 0x0c, 0xa2, 0x09, 0xc2, 0xe0, 0x4c, 0x10,
  0x86, 0x67, 0xc3, 0xb0, 0x0d, 0xc3, 0x06, 0x42, 0xc9, 0x34, 0x6e, 0x43,
  0x71, 0x61, 0x80, 0xd5, 0x55, 0x61, 0x63, 0xb3, 0x6b, 0x73, 0x49, 0x23,
  0x2b, 0x73, 0xa3, 0x9b, 0x12, 0x04, 0x55, 0xc8, 0xf0, 0x5c, 0xec, 0xca,
  0xe4, 0xe6, 0xd2, 0xde, 0xdc, 0xa6, 0x04, 0x44, 0x13, 0x32, 0x3c, 0x17,
  0xbb, 0x30, 0x36, 0xbb, 0x32, 0xb9, 0x29, 0x81, 0x51, 0x87, 0x0c, 0xcf,
  0x65, 0x0e, 0x2d, 0x8c, 0xac, 0x4c, 0xae, 0xe9, 0x8d, 0xac, 0x8c, 0x6d,
  0x4a, 0x80, 0x94, 0x21, 0xc3, 0x73, 0x91, 0x2b, 0x9b, 0x7b, 0xab, 0x93,
  0x1b, 0x2b, 0x9b, 0x9b, 0x12, 0x54, 0x75, 0xc8, 0xf0, 0x5c, 0xca, 0xdc,
  0xe8, 0xe4, 0xf2, 0xa0, 0xde, 0xd2, 0xdc, 0xe8, 0xe6, 0xa6, 0x04, 0x1d,
  0x00, 0x00, 0x00, 0x00, 0x79, 0x18, 0x00, 0x00, 0x51, 0x00, 0x00, 0x00,
  0x33, 0x08, 0x80, 0x1c, 0xc4, 0xe1, 0x1c, 0x66, 0x14, 0x01, 0x3d, 0x88,
  0x43, 0x38, 0x84, 0xc3, 0x8c, 0x42, 0x80, 0x07, 0x79, 0x78, 0x07, 0x73,
  0x98, 0x71, 0x0c, 0xe6, 0x00, 0x0f, 0xed, 0x10, 0x0e, 0xf4, 0x80, 0x0e,
  0x33, 0x0c, 0x42, 0x1e, 0xc2, 0xc1, 0x1d, 0xce, 0xa1, 0x1c, 0x66, 0x30,
  0x05, 0x3d, 0x88, 0x43, 0x38, 0x84, 0x83, 0x1b, 0xcc, 0x03, 0x3d, 0xc8,
  0x43, 0x3d, 0x8c, 0x03, 0x3d, 0xcc, 0x78, 0x8c, 0x74, 0x70, 0x07, 0x7b,
  0x08, 0x07, 0x79, 0x48, 0x87, 0x70, 0x70, 0x07, 0x7a, 0x70, 0x03, 0x76,
  0x78, 0x87, 0x70, 0x20, 0x87, 0x19, 0xcc, 0x11, 0x0e, 0xec, 0x90, 0x0e,
  0xe1, 0x30, 0x0f, 0x6e, 0x30, 0x0f, 0xe3, 0xf0, 0x0e, 0xf0, 0x50, 0x0e,
  0x33, 0x10, 0xc4, 0x1d, 0xde, 0x21, 0x1c, 0xd8, 0x21, 0x1d, 0xc2, 0x61,
  0x1e, 0x66, 0x30, 0x89, 0x3b, 0xbc, 0x83, 0x3b, 0xd0, 0x43, 0x39, 0xb4,
  0x03, 0x3c, 0xbc, 0x83, 0x3c, 0x84, 0x03, 0x3b, 0xcc, 0xf0, 0x14, 0x76,
  0x60, 0x07, 0x7b, 0x68, 0x07, 0x37, 0x68, 0x87, 0x72, 0x68, 0x07, 0x37,
  0x80, 0x87, 0x70, 0x90, 0x87, 0x70, 0x60, 0x07, 0x76, 0x28, 0x07, 0x76,
  0xf8, 0x05, 0x76, 0x78, 0x87, 0x77, 0x80, 0x87, 0x5f, 0x08, 0x87, 0x71,
  0x18, 0x87, 0x72, 0x98, 0x87, 0x79, 0x98, 0x81, 0x2c, 0xee, 0xf0, 0x0e,
  0xee, 0xe0, 0x0e, 0xf5, 0xc0, 0x0e, 0xec, 0x30, 0x03, 0x62, 0xc8, 0xa1,
  0x1c, 0xe4, 0xa1, 0x1c, 0xcc, 0xa1, 0x1c, 0xe4, 0xa1, 0x1c, 0xdc, 0x61,
  0x1c, 0xca, 0x21, 0x1c, 0xc4, 0x81, 0x1d, 0xca, 0x61, 0x06, 0xd6, 0x90,
  0x43, 0x39, 0xc8, 0x43, 0x39, 0x98, 0x43, 0x39, 0xc8, 0x43, 0x39, 0xb8,
  0xc3, 0x38, 0x94, 0x43, 0x38, 0x88, 0x03, 0x3b, 0x94, 0xc3, 0x2f, 0xbc,
  0x83, 0x3c, 0xfc, 0x82, 0x3b, 0xd4, 0x03, 0x3b, 0xb0, 0xc3, 0x0c, 0xc4,
  0x21, 0x07, 0x7c, 0x70, 0x03, 0x7a, 0x28, 0x87, 0x76, 0x80, 0x87, 0x19,
  0xd1, 0x43, 0x0e, 0xf8, 0xe0, 0x06, 0xe4, 0x20, 0x0e, 0xe7, 0xe0, 0x06,
  0xf6, 0x10, 0x0e, 0xf2, 0xc0, 0x0e, 0xe1, 0x90, 0x0f, 0xef, 0x50, 0x0f,
  0xf4, 0x30, 0x83, 0x81, 0xc8, 0x01, 0x1f, 0xdc, 0x40, 0x1c, 0xe4, 0xa1,
  0x1c, 0xc2, 0x61, 0x1d, 0xdc, 0x40, 0x1c, 0xe4, 0x01, 0x00, 0x00, 0x00,
  0x71, 0x20, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x06, 0x30, 0x74, 0x5f,
  0x6b, 0x06, 0xdb, 0x70, 0xf9, 0xce, 0xe3, 0x0b, 0x01, 0x55, 0x14, 0x44,
  0x54, 0x3a, 0xc0, 0x50, 0x12, 0x06, 0x20, 0x60, 0x7e, 0x71, 0xdb, 0x56,
  0xb0, 0x0d, 0x97, 0xef, 0x3c, 0xbe, 0x10, 0x50, 0x45, 0x41, 0x44, 0xa5,
  0x03, 0x0c, 0x25, 0x61, 0x00, 0x02, 0xe6, 0x23, 0xb7, 0x6d, 0x07, 0xd2,
  0x70, 0xf9, 0xce, 0xe3, 0x0b, 0x11, 0x01, 0x4c, 0x44, 0x08, 0x34, 0xc3,
  0x42, 0xd8, 0xc0, 0x35, 0x5c, 0xbe, 0xf3, 0xf8, 0x11, 0x60, 0x6d, 0x54,
  0x51, 0x10, 0x51, 0xe9, 0x00, 0x83, 0x5f, 0xd4, 0xba, 0x11, 0x60, 0xc3,
  0xe5, 0x3b, 0x8f, 0x1f, 0x01, 0xd6, 0x46, 0x15, 0x05, 0x11, 0xb1, 0x93,
  0x13, 0x11, 0x7e, 0x51, 0xeb, 0x16, 0x20, 0x0d, 0x97, 0xef, 0x3c, 0xfe,
  0x74, 0x44, 0x04, 0x30, 0x88, 0x83, 0x8f, 0xdc, 0xb6, 0x09, 0x3c, 0xc3,
  0xe5, 0x3b, 0x8f, 0x4f, 0x35, 0x40, 0x84, 0xf9, 0xc5, 0x6d, 0x03, 0x00,
  0x61, 0x20, 0x00, 0x00, 0x9c, 0x00, 0x00, 0x00, 0x13, 0x04, 0x47, 0x2c,
  0x10, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x34, 0x94, 0x5d, 0x59,
  0x0a, 0x94, 0x5c, 0x29, 0x94, 0x4f, 0x0d, 0x14, 0xa6, 0x40, 0xe9, 0x06,
  0x94, 0x41, 0x69, 0xd0, 0x52, 0x04, 0x25, 0x40, 0xc6, 0x18, 0xc1, 0xee,
  0x8f, 0x32, 0x0b, 0x0e, 0x94, 0xcc, 0x00, 0x90, 0x31, 0x46, 0xb0, 0xfb,
  0xa3, 0xcc, 0x82, 0xc1, 0x08, 0x00, 0x00, 0x00, 0x23, 0x06, 0x09, 0x00,
  0x82, 0x60, 0x90, 0x75, 0x4e, 0xc1, 0x71, 0xd4, 0x88, 0x41, 0x02, 0x80,
  0x20, 0x18, 0x64, 0xde, 0x63, 0x68, 0x5a, 0x35, 0x62, 0x90, 0x00, 0x20,
  0x08, 0x06, 0xd9, 0x07, 0x21, 0xdb, 0x66, 0x8d, 0x18, 0x18, 0x00, 0x08,
  0x82, 0x01, 0x61, 0x06, 0x0c, 0x37, 0x62, 0x70, 0x00, 0x20, 0x08, 0x06,
  0xd3, 0x18, 0x50, 0x42, 0x37, 0x9a, 0x10, 0x00, 0x15, 0x0c, 0x30, 0x9a,
  0x30, 0x04, 0xc3, 0x0d, 0x42, 0x40, 0x06, 0xb3, 0x0c, 0x81, 0x11, 0x8c,
  0x18, 0x1c, 0x00, 0x08, 0x82, 0xc1, 0x84, 0x06, 0xd9, 0x61, 0x8d, 0x26,
  0x04, 0x41, 0x05, 0x67, 0x80, 0xa3, 0x09, 0x88, 0x50, 0x41, 0xa6, 0xa5,
  0x06, 0xc1, 0xd5, 0xb0, 0x41, 0x05, 0x9c, 0x5a, 0x1b, 0x04, 0x17, 0x18,
  0x31, 0x38, 0x00, 0x10, 0x04, 0x83, 0x29, 0x0e, 0xc4, 0x00, 0xd2, 0x46,
  0x13, 0x82, 0x60, 0x34, 0x41, 0x10, 0x2a, 0x10, 0xa4, 0xa0, 0xa0, 0x2a,
  0x12, 0xa6, 0x04, 0x62, 0x6a, 0x28, 0xaa, 0x84, 0x06, 0x2b, 0x58, 0xae,
  0x96, 0x33, 0x80, 0x2a, 0x02, 0xad, 0x21, 0x80, 0x0a, 0x28, 0x18, 0x31,
  0x38, 0x00, 0x10, 0x04, 0x83, 0xe9, 0x0f, 0xe0, 0xc0, 0xdb, 0x83, 0xd1,
  0x84, 0x00, 0xa8, 0x60, 0x91, 0xd1, 0x84, 0x21, 0x28, 0x23, 0x90, 0xd1,
  0x84, 0x42, 0xa8, 0xa0, 0x91, 0x0a, 0x0a, 0xa8, 0x80, 0x80, 0x11, 0x03,
  0x05, 0x00, 0x41, 0x30, 0x70, 0x54, 0x01, 0x0f, 0xd0, 0x20, 0x10, 0x05,
  0x37, 0x20, 0x85, 0xd1, 0x84, 0x00, 0xb8, 0xc0, 0xc0, 0x11, 0x83, 0x03,
  0x00, 0x41, 0x30, 0x98, 0x56, 0x81, 0x0f, 0xd4, 0x00, 0x15, 0x46, 0x13,
  0x82, 0x61, 0xb8, 0x21, 0x48, 0x05, 0x30, 0x98, 0x65, 0x10, 0x86, 0x60,
  0x34, 0xe1, 0x19, 0x2a, 0x40, 0x60, 0xc4, 0x40, 0x01, 0x40, 0x10, 0x0c,
  0x1c, 0x59, 0x00, 0x05, 0x38, 0x08, 0x54, 0xc1, 0x0e, 0x58, 0x61, 0x34,
  0x21, 0x00, 0x2e, 0x30, 0x70, 0x96, 0x60, 0x18, 0xa8, 0x30, 0x04, 0x41,
  0x1d, 0x82, 0x92, 0x2c, 0xa9, 0x80, 0x82, 0x0a, 0x22, 0x18, 0x31, 0x50,
  0x00, 0x10, 0x04, 0x03, 0x27, 0x17, 0x4e, 0xe1, 0x0e, 0x82, 0x58, 0xe8,
  0x83, 0x59, 0x18, 0x4d, 0x08, 0x80, 0x0b, 0x0c, 0x9c, 0x65, 0x20, 0x8a,
  0x66, 0x34, 0x61, 0x1b, 0x2a, 0x28, 0x60, 0xc4, 0x40, 0x01, 0x40, 0x10,
  0x0c, 0x1c, 0x5f, 0x60, 0x05, 0x3e, 0x08, 0x6c, 0x41, 0x14, 0x70, 0x61,
  0x34, 0x21, 0x00, 0x2e, 0x30, 0x70, 0x96, 0xa0, 0x18, 0xa8, 0x30, 0x04,
  0x42, 0x24, 0x86, 0x9a, 0x03, 0x56, 0x80, 0x52, 0x83, 0x40, 0x47, 0x13,
  0xfe, 0x60, 0x18, 0x6e, 0x08, 0xc0, 0x01, 0x0c, 0xa6, 0x1b, 0x4e, 0x21,
  0x15, 0x82, 0x23, 0x8c, 0x32, 0x21, 0x90, 0xcf, 0xe9, 0x81, 0x51, 0x26,
  0x04, 0xf4, 0x19, 0x31, 0x30, 0x00, 0x10, 0x04, 0x83, 0x83, 0x1d, 0xc4,
  0x21, 0x18, 0x31, 0x30, 0x00, 0x10, 0x04, 0x83, 0xa3, 0x1d, 0x64, 0x41,
  0x18, 0x31, 0x38, 0x00, 0x10, 0x04, 0x83, 0x69, 0x1d, 0x78, 0x41, 0x15,
  0xc4, 0x61, 0x34, 0x21, 0x10, 0x2a, 0x40, 0x05, 0x19, 0x4d, 0x18, 0x86,
  0x12, 0x02, 0x18, 0x31, 0x38, 0x00, 0x10, 0x04, 0x03, 0xeb, 0x1d, 0xc2,
  0xe1, 0x15, 0x7c, 0x61, 0x34, 0x21, 0x00, 0x2c, 0xc9, 0xe4, 0x63, 0x09,
  0x25, 0x1f, 0x13, 0xd4, 0x00, 0x3e, 0x16, 0x08, 0xf1, 0xb1, 0x22, 0x90,
  0xcf, 0x05, 0xc9, 0x8d, 0x18, 0x38, 0x00, 0x08, 0x82, 0x01, 0x94, 0x0f,
  0xe3, 0x90, 0x0b, 0x8b, 0x3c, 0x04, 0xbc, 0xc0, 0x0b, 0xbc, 0xe0, 0x0b,
  0xf4, 0x30, 0x62, 0x70, 0x00, 0x20, 0x08, 0x06, 0xd6, 0x3d, 0xa4, 0xc3,
  0x2d, 0x98, 0xc3, 0x68, 0x42, 0x00, 0x58, 0xb4, 0xc9, 0xc7, 0x22, 0x31,
  0x90, 0x8f, 0x09, 0x69, 0x00, 0x1f, 0x0b, 0x04, 0xf8, 0x58, 0x11, 0xc8,
  0xe7, 0x82, 0xe4, 0x46, 0x0c, 0x1c, 0x00, 0x04, 0xc1, 0x00, 0x0a, 0x89,
  0x75, 0x08, 0x87, 0x48, 0x1f, 0x02, 0x72, 0x20, 0x07, 0x72, 0x30, 0x07,
  0x7e, 0x98, 0x25, 0x30, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
