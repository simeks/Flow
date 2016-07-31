#include <Core/Common.h>
#include <Core/Flow/FlowObject.h>
#include <Core/Flow/Field.h>

#include <Tools/Testing/Framework.h>

using namespace testing;

class TestObject : public FlowObject
{
    DECLARE_OBJECT(TestObject, FlowObject);
public:
    TestObject() : a(1), b(2.0), c("3") {}

    int a;
    float b;
    std::string c;

    static void populate_class(FlowClass& c);
};

class TestObject2 : public TestObject
{
    DECLARE_OBJECT(TestObject2, TestObject);
public:
    TestObject2() : d(4) {}

    int d;

    static void populate_class(FlowClass& c);
};

void TestObject::populate_class(FlowClass& c)
{
    c.add_field<Int32Field>("a", offsetof(TestObject, TestObject::a), 0);
    c.add_field<Float32Field>("b", offsetof(TestObject, TestObject::b), 0);
    c.add_field<StringField>("c", offsetof(TestObject, TestObject::c), 0);
}

void TestObject2::populate_class(FlowClass& c)
{
    c.add_field<Int32Field>("d", offsetof(TestObject2, TestObject2::d), 0);
}
IMPLEMENT_OBJECT2(TestObject, "TestObject", &TestObject::populate_class);
IMPLEMENT_OBJECT2(TestObject2, "TestObject2", &TestObject2::populate_class);

TEST_CASE(FlowObject_find_field)
{
    Field* f = TestObject::static_class()->find_field("a");
    ASSERT_EXPR(f != nullptr);
    ASSERT_EXPR(f->is_a(Int32Field::static_class()));

    f = TestObject::static_class()->find_field("b");
    ASSERT_EXPR(f != nullptr);
    ASSERT_EXPR(f->is_a(Float32Field::static_class()));

    f = TestObject::static_class()->find_field("c");
    ASSERT_EXPR(f != nullptr);
    ASSERT_EXPR(f->is_a(StringField::static_class()));
}


TEST_CASE(FlowObject_field_value)
{
    TestObject* obj = new TestObject();

    Field* f = TestObject::static_class()->find_field("a");
    ASSERT_EXPR(f != nullptr);
    ASSERT_EQUAL(obj->field_value((Int32Field*)f), obj->a);

    f = TestObject::static_class()->find_field("b");
    ASSERT_EXPR(f != nullptr);
    ASSERT_EQUAL_F(obj->field_value((Float32Field*)f), obj->b, FLT_EPSILON);

    f = TestObject::static_class()->find_field("c");
    ASSERT_EXPR(f != nullptr);
    ASSERT_EQUAL(obj->field_value((StringField*)f), obj->c);
}

TEST_CASE(FlowObject_set_field_value)
{
    TestObject* obj = new TestObject();

    Field* f = TestObject::static_class()->find_field("a");
    ASSERT_EXPR(f != nullptr);
    obj->set_field_value((Int32Field*)f, 99);
    ASSERT_EQUAL(obj->a, 99);

    f = TestObject::static_class()->find_field("b");
    ASSERT_EXPR(f != nullptr);
    obj->set_field_value((Float32Field*)f, 88.8f);
    ASSERT_EQUAL_F(obj->b, 88.8f, FLT_EPSILON);

    f = TestObject::static_class()->find_field("c");
    ASSERT_EXPR(f != nullptr);
    obj->set_field_value((StringField*)f, "test");
    ASSERT_EQUAL(obj->c, std::string("test"));
}

TEST_CASE(FlowObject_inheritance)
{
    TestObject* obj2 = new TestObject2();

    Field* f = TestObject2::static_class()->find_field("d");
    ASSERT_EXPR(f != nullptr);
    ASSERT_EXPR(f->is_a(Int32Field::static_class()));
    ASSERT_EQUAL(obj2->field_value((Int32Field*)f), ((TestObject2*)obj2)->d);

    obj2->set_field_value((Int32Field*)f, 99);
    ASSERT_EQUAL(((TestObject2*)obj2)->d, 99);

    f = TestObject2::static_class()->find_field("a");
    ASSERT_EXPR(f != nullptr);
    ASSERT_EXPR(f->is_a(Int32Field::static_class()));
    ASSERT_EQUAL(obj2->field_value((Int32Field*)f), ((TestObject2*)obj2)->a);

    obj2->set_field_value((Int32Field*)f, 11);
    ASSERT_EQUAL(((TestObject2*)obj2)->a, 11);

}
