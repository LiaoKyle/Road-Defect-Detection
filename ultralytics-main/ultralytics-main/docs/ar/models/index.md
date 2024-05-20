---
comments: true
description: استكشف مجموعة متنوعة من عائلة YOLO، SAM، MobileSAM، FastSAM، YOLO-NAS ونماذج RT-DETR المدعومة بواسطة Ultralytics. ابدأ مع أمثلة للإستخدام باستخدام واجهة سطر الأوامر ولغة البايثون.
keywords: Ultralytics، مستندات، YOLO، SAM، MobileSAM، FastSAM، YOLO-NAS، RT-DETR، نماذج، هندسات، Python، CLI
---

# النماذج المدعومة بواسطة Ultralytics

مرحبًا بك في مستندات نماذج Ultralytics! نحن نقدم دعمًا لمجموعة واسعة من النماذج، وكل نموذج مصمم لمهام محددة مثل [كشف الكائنات](../tasks/detect.md)، [تجزئة الحالات](../tasks/segment.md)، [تصنيف الصور](../tasks/classify.md)، [تقدير الوضع](../tasks/pose.md)، و[تتبع العديد من الكائنات](../modes/track.md). إذا كنت مهتمًا بالمساهمة في بنية نموذجك في Ultralytics ، تحقق من [دليل المساهمة](../../help/contributing.md).

!!! Note "ملاحظة"

    🚧 مستنداتنا متعددة اللغات قيد الإنشاء حاليًا ونحن نعمل بجد لتحسينها. شكرا لك على صبرك! 🙏

## النماذج المميزة

فيما يلي بعض النماذج الرئيسية المدعمة:

1. **[YOLOv3](../../models/yolov3.md)**: الإصدار الثالث من عائلة نموذج YOLO، والذي أنشأه جوزيف ريدمون، والمعروف بقدرته على الكشف في الوقت الحقيقي بكفاءة.
2. **[YOLOv4](../../models/yolov4.md)**: تحديث لنموذج YOLOv3 الأصلي من قبل اليكسي بوتشكوفسكي في عام 2020.
3. **[YOLOv5](../../models/yolov5.md)**: إصدار محسن لبنية YOLO بواسطة Ultralytics ، يقدم أداءً أفضل وتفاوتات سرعة مقارنة بالإصدارات السابقة.
4. **[YOLOv6](../../models/yolov6.md)**: تم إصداره بواسطة [ميتوان](https://about.meituan.com/) في عام 2022 ، ويستخدم في العديد من روبوتات التسليم الذاتي للشركة.
5. **[YOLOv7](../../models/yolov7.md)**: نماذج YOLO المحدثة التي تم إطلاقها في عام 2022 من قبل أصحاب YOLOv4.
6. **[YOLOv8](../../models/yolov8.md)**: أحدث إصدار من عائلة YOLO ، ويتميز بقدرات محسنة مثل تجزئة الحالات، وتقدير النقاط الرئيسة، والتصنيف.
7. **[Segment Anything Model (SAM)](../../models/sam.md)**: نموذج Segment Anything Model (SAM) من Meta.
8. **[Mobile Segment Anything Model (MobileSAM)](../../models/mobile-sam.md)**: MobileSAM لتطبيقات الهواتف المحمولة ، من جامعة Kyung Hee.
9. **[Fast Segment Anything Model (FastSAM)](../../models/fast-sam.md)**: FastSAM من مجموعة تحليل الصور والفيديو، معهد الأتمتة، الأكاديمية الصينية للعلوم.
10. **[YOLO-NAS](../../models/yolo-nas.md)**: نماذج YOLO للبحث في تصميم العمارة العصبية.
11. **[Realtime Detection Transformers (RT-DETR)](../../models/rtdetr.md)**: نماذج PaddlePaddle Realtime Detection Transformer (RT-DETR) من Baidu.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/MWq1UxqTClU?si=nHAW-lYDzrz68jR0"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>شاهد:</strong> قم بتشغيل نماذج Ultralytics YOLO في بضعة أسطر من الكود.
</p>

## البدء: أمثلة للإستخدام

!!! Example "مثال"

    === "Python"

        يمكن تمرير نماذج PyTorch مدربة سابقًا بتنسيق `*.pt` بالإضافة إلى ملفات التكوين بتنسيق  `*.yaml` إلى  فئات `YOLO()`، `SAM()`، `NAS()` و `RTDETR()` لإنشاء نموذج في Python:

        ```python
        from ultralytics import YOLO

        # قم بتحميل نموذج YOLOv8n المدرب بـ COCO المسبق
        model = YOLO('yolov8n.pt')

        # عرض معلومات النموذج (اختياري)
        model.info()

        # قم بتدريب النموذج على مجموعة بيانات COCO8 لمدة 100 حقبة
        results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

        # قم بتشغيل الاستدلال مع نموذج YOLOv8n على صورة 'bus.jpg'
        results = model('path/to/bus.jpg')
        ```
    === "CLI"

        هناك أوامر CLI متاحة لتشغيل النماذج مباشرةً:

        ```bash
        # قم بتحميل نموذج YOLOv8n المدرب بـ COCO المسبق و تدريبه على مجموعة بيانات COCO8 لمدة 100 حقبة
        yolo train model=yolov8n.pt data=coco8.yaml epochs=100 imgsz=640

        # قم بتحميل نموذج YOLOv8n المدرب بـ COCO المسبق و قم بتشغيل الاستدلال على صورة 'bus.jpg'
        yolo predict model=yolov8n.pt source=path/to/bus.jpg
        ```

## المساهمة في نماذج جديدة

هل ترغب في المساهمة بنموذجك في Ultralytics؟ رائع! نحن مفتوحون دائمًا لتوسيع مجموعة النماذج الخاصة بنا.

1. **انسخ المستودع**: ابدأ بإنشاء فرع جديد في مستودع [Ultralytics GitHub repository ](https://github.com/ultralytics/ultralytics).

2. **نسخ Fork الخاص بك**: نسخ Fork الخاص بك إلى جهاز الكمبيوتر المحلي الخاص بك وأنشئ فرعًا جديدًا للعمل عليه.

3. **اتبع نموذجك**: قم بإضافة نموذجك وفقًا لمعايير البرمجة والتوجيهات المقدمة في [دليل المساهمة](../../help/contributing.md).

4. **اختبر بدقة**: تأكد من اختبار نموذجك بدقة ، سواء بشكل منفصل أم كجزء من السلسلة.

5. **أنشئ طلبًا للدمج**: بمجرد أن تكون راضيًا عن نموذجك، قم بإنشاء طلب للدمج إلى البرنامج الرئيسي للمراجعة.

6. **استعراض ودمج الكود**: بعد المراجعة، إذا كان نموذجك يلبي معاييرنا، فسيتم دمجه في البرنامج الرئيسي.

للخطوات المفصلة ، استشر [دليل المساهمة](../../help/contributing.md).
