import {
    Block,
    Div,
    PointerPosition,
    Api,
    Form,
    SimpleTextInput,
    SimpleSelect,
    TasksColumn,
    Task,
    TaskData,
    Tools,
    ClientLocation,
    Listener,
    AddTaskPopup,
    Button,
    WebSocketClient,
    WebSocketMessage
} from '@src/classes';

export type TaskStatus = 'PREDEFINED' | 'TODO' | 'IN_PROGRESS' | 'DONE' | 'ARCHIVED';

export class TasksKanban extends Div {

    static readonly DISPLAYED_STATUSES: TaskStatus[] = ['PREDEFINED', 'TODO', 'IN_PROGRESS', 'DONE'];

    private client: WebSocketClient;

    private filtersContainer: Div;
    private kanbanContainer: Div;
    
    private columns: Map<TaskStatus, TasksColumn> = new Map();

    private grabbedTask: Task | null = null;
    private grabbedClonedTask: Task | null = null;
    private grabbedClonedTaskPointerOffsetX: number;
    private grabbedClonedTaskPointerOffsetY: number;

    private mouseMoveListener: Listener;
    private mouseUpListener: Listener;
    private resizeListener: Listener;
    private rect: DOMRect;
    private hoveredColumn: TasksColumn | null = null;

    private searchInput: SimpleTextInput;
    private categorySelect: SimpleSelect;
    private teamMemberSelect: SimpleSelect;

    private resetButton: Button;

    private updateColumnsInterval: ReturnType<typeof setInterval>

    constructor(private teamId: number, parent: Block) {

        super('tasks-kanban', parent);

        this.initWebSocket();

        this.drawFilters();
        this.drawKanban();

        this.mouseMoveListener = ClientLocation.get().on('mouse-move', this.onMouseMove.bind(this));
        this.mouseUpListener = ClientLocation.get().on('mouse-up', this.onMouseUp.bind(this));
        this.resizeListener = ClientLocation.get().on('resize', this.onResize.bind(this));

        this.updateColumnsInterval = setInterval(this.updateColumns.bind(this), 60_000);
    }
    
    /*
    **
    **
    */
    private async initWebSocket() : Promise<void> {

        this.client = new WebSocketClient('/tasks/ws');

        this.client.on('open', this.onWebSocketOpen.bind(this));

        await this.client.open();
    }

    /*
    **
    **
    */
    private async onWebSocketOpen() : Promise<void> {

        await this.client.sendRequest('/team/join', {
            teamId: this.teamId
        });
    
        this.client?.listen('/task/sync', this.onTaskSyncMessage.bind(this));
    }

    /*
    **
    **
    */
    public onTaskSyncMessage(message: WebSocketMessage) : void {

        this.syncTask(message.data.taskId);
    }

    /*
    **
    **
    */
    private async drawFilters() : Promise<void> {

        if (!this.filtersContainer)
            this.filtersContainer = new Div('filters-container', this);

        const form = new Form(this.filtersContainer);

        //============
        //SEARCH INPUT
        //============

        this.searchInput = new SimpleTextInput({
            label: 'Search'
        }, form);
        
        this.searchInput.on('value', this.filter.bind(this));

        //==============
        //CATEGORY INPUT
        //==============

        this.categorySelect = new SimpleSelect({
            label: 'Category',
            value: 'all',
            items: await this.getTeamCategories()
        }, form);

        this.categorySelect.on('value', this.filter.bind(this));

        //=================
        //TEAM MEMBER INPUT
        //=================

        this.teamMemberSelect = new SimpleSelect({
            label: 'Team member',
            value: 'all',
            items: await this.getTeamAccounts()
        }, form);

        this.teamMemberSelect.on('value', this.filter.bind(this));

        //============
        //RESET BUTTON
        //============

        this.resetButton = new Button({
            label: 'Reset'
        }, form).onNative('click', this.resetFilters.bind(this));

        await Tools.sleep(10);

        this.computeResetButtonState();
        
        this.filtersContainer.setData('displayed', 1);
    }

    /*
    **
    **
    */
    private filter() {

        const text = this.searchInput.getValue().trim() === '' ? null : this.searchInput.getValue().toLowerCase();
        const categoryId = this.categorySelect.getValue() === 'all' ? null : parseInt(this.categorySelect.getValue() as string);
        const accountId = this.teamMemberSelect.getValue() === 'all' ? null : parseInt(this.teamMemberSelect.getValue() as string);

        for (const [status, column] of this.columns.entries())
            column.filter(text, categoryId, accountId);

        this.computeResetButtonState();
    }

    /*
    **
    **
    */
    private resetFilters() : void {

        this.searchInput.setValue('', false);
        this.categorySelect.setValue('all', false);
        this.teamMemberSelect.setValue('all', false);

        this.computeResetButtonState();
        this.filter();
    }

    /*
    **
    **
    */
    private computeResetButtonState() : void  {

        if (this.searchInput.getValue() === '' && this.categorySelect.getValue() === 'all' && this.teamMemberSelect.getValue() === 'all')
            this.resetButton.disable();
        else
            this.resetButton.enable();
    }

    /*
    **
    **
    */
    private async drawKanban() : Promise<void> {
        
        if (!this.kanbanContainer)
            this.kanbanContainer = new Div('kanban-container', this);

        const tasks = await Api.get('/team/tasks', {
            teamId: this.teamId
        });

        for (const status of TasksKanban.DISPLAYED_STATUSES) {

            const tasksColumn = new TasksColumn(status, this, this.kanbanContainer);

            tasksColumn.on('grabbed-task', this.onGrabbedTask.bind(this));

            this.columns.set(status, tasksColumn);
        }

        for (const data of tasks) {
            
            const column = this.columns.get(data.status);

            if (!column)
                continue;
            
            column.instanciateTask(data);
        }

        await Tools.sleep(10);

        this.kanbanContainer.setData('displayed', 1);
    }

    /*
    **
    **
    */
    public getTeamId() : number {

        return this.teamId;
    }

    /*
    **
    **
    */
    private async moveTask(id: number, status: TaskStatus) : Promise<void> {

        await Api.post('/task/move', {
            id: id,
            status: status
        });
    }
    
    /*
    **
    **
    */
    private async getTeamCategories() : Promise<any[]> {
        
        let output : any[] = [{label: 'All', value: 'all'}];

        let data = await Api.get('/team/categories', {
            teamId: this.teamId
        });
        
        for (let row of data) {
            output.push({
                label: row.title,
                value: row.id
            });
        }

        return output;
    }

    /*
    **
    **
    */
    private async getTeamAccounts() : Promise<any[]> {
        
        let output : any[] = [{label: 'All', value: 'all'}];

        let data = await Api.get('/team/accounts', {
            teamId: this.teamId
        });

        for (let row of data) {
            output.push({
                label: `${row.last_name} ${row.first_name}`,
                value: row.id
            });
        }

        return output;
    }

    /*
    **
    **
    */
    public async syncTask(id: number) : Promise<void> {

        //======================================
        //DELETE EXISTING DRAWN TASK (IF EXISTS)
        //======================================

        for (const [status, column] of this.columns.entries()) {

            for (const task of column.getTasks()) {
            
                if (task.getData().id === id) {
                    
                    column.deleteTask(task);
                    break;
                }
            }
        }

        //===============================================================================
        //DOWNLOAD AND GENERATE FRESH TASK (IF EXISTS) IN APPRORPRIATE COLUMN (IF EXISTS)
        //===============================================================================

        try {
            
            const data: TaskData = await Api.get('/task', {
                id: id
            });
            
            const targetColumn = this.columns.get(data.status as TaskStatus);

            if (!targetColumn)
                return;

            targetColumn.instanciateTask(data);
        }
        catch(ignored) {
        }
    }

    /*
    **
    **
    */
    private onResize() {

        this.rect = this.kanbanContainer.element.getBoundingClientRect();
    }

    /*
    **
    **
    */
    private onGrabbedTask(task: Task) : void {
        
        this.grabbedTask = task;

        this.grabbedTask.setTranslucent();

        this.grabbedClonedTask = new Task(task.getData(), task.getColumn());
        this.grabbedClonedTask.setData('grabbed-cloned', 1);

        const rect = this.grabbedTask.element.getBoundingClientRect();

        this.grabbedClonedTask.setStyles({
            left: `${rect.x}px`,
            top: `${rect.y}px`,
            width: `${rect.width}px`
        });

        this.grabbedClonedTaskPointerOffsetX = ClientLocation.get().pointer.x - rect.x;
        this.grabbedClonedTaskPointerOffsetY = ClientLocation.get().pointer.y - rect.y;

        ClientLocation.get().block.append(this.grabbedClonedTask);

        this.rect = this.kanbanContainer.element.getBoundingClientRect();

        for (const [status, column] of this.columns.entries())
            column.recordRect();
    }

    /*
    **
    **
    */
    private onMouseMove(pointer: PointerPosition) : void {

        if (!this.grabbedClonedTask)
            return;

        if (this.isPointerOutside()) {

            this.grabbedTask?.display();
            this.grabbedTask = null;
            
            this.deleteGrabbedClonedTask();
            
            this.hoveredColumn?.setPotentialDropStatus(0);

            return;
        }

        this.grabbedClonedTask.setStyles({
            left: `${pointer.x - this.grabbedClonedTaskPointerOffsetX}px`,
            top: `${pointer.y - this.grabbedClonedTaskPointerOffsetY}px`
        });

        this.populateHoveredColumn();

        const taskStatus = this.grabbedTask?.getColumn().getStatus();

        if (taskStatus === this.hoveredColumn?.getStatus())
            return;

        switch (this.hoveredColumn?.getStatus()) {

            case 'PREDEFINED': {
                
                this.hoveredColumn.setPotentialDropStatus(0);
                
                break;
            }

            case 'TODO': {

                this.hoveredColumn.setPotentialDropStatus(taskStatus === 'PREDEFINED' ? 1 : 0);

                break;
            }

            case 'IN_PROGRESS': {

                this.hoveredColumn.setPotentialDropStatus(taskStatus === 'TODO' ? 1 : 0);

                break;
            }

            case 'DONE': {

                this.hoveredColumn.setPotentialDropStatus(taskStatus === 'IN_PROGRESS' ? 1 : 0);

                break;
            }
        }
    }

    /*
    **
    **
    */
    private onMouseUp() : void {
        
        if (!this.grabbedClonedTask || !this.grabbedTask)
            return;

        this.deleteGrabbedClonedTask();
        this.hoveredColumn?.setPotentialDropStatus(0);
        
        const taskStatus = this.grabbedTask.getColumn().getStatus();
        
        if (taskStatus === this.hoveredColumn?.getStatus()) {
            this.grabbedTask.display();
            return;
        }

        switch (this.hoveredColumn?.getStatus()) {

            case 'TODO': {

                if (taskStatus === 'PREDEFINED') {

                    const popup = new AddTaskPopup(this.teamId, 'TODO', this.grabbedTask.getData())
                    
                    popup.on('hide', () => {
                        this.grabbedTask?.display();
                    });
                }
                else
                    this.grabbedTask.display();

                break;
            }

            case 'IN_PROGRESS': {

                if (taskStatus === 'TODO')
                    this.moveTask(this.grabbedTask.getData().id, 'IN_PROGRESS');
                else
                    this.grabbedTask.display();

                break;
            }

            case 'DONE': {

                if (taskStatus === 'IN_PROGRESS')
                    this.moveTask(this.grabbedTask.getData().id, 'DONE');
                else
                    this.grabbedTask.display();

                break;
            }

            default: this.grabbedTask.display();
        }
    }

    /*
    **
    **
    */
    private deleteGrabbedClonedTask() : void {

        this.grabbedClonedTask?.delete();
        this.grabbedClonedTask = null;
    }

    /*
    **
    **
    */
    public getGrabbedTask() : Task | null {

        return this.grabbedTask;
    }

    /*
    **
    **
    */
    public getGrabbedClonedTask() : Task | null {

        return this.grabbedClonedTask;
    }

    /*
    **
    **
    */
    private isPointerOutside() : boolean {

        const pointer = ClientLocation.get().pointer;

        return (
            pointer.x < this.rect.left ||
            pointer.x >= this.rect.right ||
            pointer.y < this.rect.top ||
            pointer.y >= this.rect.bottom
        );
    }
    
    /*
    **
    **
    */
    private populateHoveredColumn() : void {

        let hoveredColumn: TasksColumn | null = null;

        for (const [status, column] of this.columns.entries()) {
            
            if (column.isPointerInside()) {
                hoveredColumn = column;
                break;
            }
        }
        
        if (this.hoveredColumn) {
        
            if (!hoveredColumn || (this.hoveredColumn.getStatus() !== hoveredColumn.getStatus()))
                this.hoveredColumn.setPotentialDropStatus(0);
        }

        this.hoveredColumn = hoveredColumn;
    }

    /*
    **
    **
    */
    private updateColumns() : void {

        for (const [status, column] of this.columns.entries())
            column.updateTasks();       
    }

    /*
    **
    **
    */
    public release() : void {

        this.client?.close();

        this.mouseMoveListener.off();
        this.mouseUpListener.off();
        this.resizeListener.off();

        clearInterval(this.updateColumnsInterval);
    }

    /*
    **
    **
    */
    public onBeforeDelete(): void {
        
        this.release();
    }
}