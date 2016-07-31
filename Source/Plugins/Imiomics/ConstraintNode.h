#ifndef __IMIOMICS_CONSTRAINT_NODE_H__
#define __IMIOMICS_CONSTRAINT_NODE_H__

#include <Core/Flow/FlowNode.h>

class SubjectDatabase;
class ConstraintNode : public FlowNode
{
    DECLARE_OBJECT(ConstraintNode, FlowNode);
public:
    ConstraintNode(SubjectDatabase* db = nullptr);
    virtual ~ConstraintNode();

    virtual void run(FlowContext& context) OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;

private:
    SubjectDatabase* _subject_db;
};

class ConstraintPathsNode : public FlowNode
{
    DECLARE_OBJECT(ConstraintPathsNode, FlowNode);
public:
    ConstraintPathsNode(SubjectDatabase* db = nullptr);
    virtual ~ConstraintPathsNode();

    virtual void run(FlowContext& context) OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;

private:
    SubjectDatabase* _subject_db;
};

#endif // __IMIOMICS_CONSTRAINT_NODE_H__
